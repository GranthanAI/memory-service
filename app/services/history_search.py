"""
app/services/history_search.py

Question-aware retrieval of OLDER ORIGINAL MESSAGES of the same conversation.

A summary is lossy by design. When the current question needs something that happened earlier and may not be in it (an
exact wording, an old decision's reason, a specific error text, an identifier), the original message is recovered from the
durable `conversation_recent_messages` rows, which this service keeps for every message of the conversation.

No new index or service: a bounded single-partition read (HISTORY_SEARCH_MAX_MESSAGES newest rows) is scored lexically.
Lexical matching is deliberate: the cases that matter here (error strings, identifiers, quoted names, "what did I say
about X") are exact-term cases, and it needs no embeddings of every message.

Safety
  * Only THIS conversation's own partition is read: a sibling or parent branch can never contribute a message. Ancestor
    context stays the branch-valid summaries (and the inherited turns conversation-service already bounds); ownership is
    checked by the caller before this runs.
  * Messages already in the verbatim context (the recent window / tail / unsummarised ones) are never repeated.
  * A hit is returned only when enough of the question's DISTINCTIVE terms are in it. If nothing qualifies, nothing is
    returned and the caller says so; this never invents or "fills in" history.
"""

import asyncio
import math
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from app.core.config import SystemSettings, settings as default_settings
from app.core.metrics import HISTORY_SEARCH, HISTORY_SEARCH_SECONDS
from app.services.summary_service import message_key
from app.utils.token_counter import TokenCounter

STOPWORDS = frozenset("""
a about above after again all also am an and any are as at be because been before being below between both but by can
could did do does doing down during each few for from further had has have having he her here hers him his how i if in into
is it its just me more most my no nor not now of off on once only or other our out over own same she should so some such
than that the their them then there these they this those through to too under until up very was we were what when where
which while who whom why will with would you your yours please tell give show make let get use using need want like
""".split())
RECALL_CUES = re.compile(
    r"\b(earlier|before|previously|previous|last time|originally|at the start|at the beginning|back when|"
    r"we (?:decided|agreed|discussed|said|talked)|you (?:said|told|mentioned|suggested|wrote)|i (?:said|told|mentioned|wrote)|"
    r"remind me|recall|remember|exact(?:ly)?|verbatim|word for word|the error|stack ?trace|that (?:message|error|code|name))\b",
    re.IGNORECASE)
_WORD = re.compile(r"\w[\w.\-/:]*\w|\w", re.UNICODE)
MIN_CONTENT_CHARS = 15


def _terms(text: str) -> List[str]:
    out: List[str] = []
    for raw in _WORD.findall((text or "").lower()):
        parts = [raw] + ([p for p in re.split(r"[.\-/:_]", raw) if len(p) >= 3] if re.search(r"[.\-/:_]", raw) else [])
        for tok in parts:
            if tok in STOPWORDS or (tok.isdigit() and len(tok) < 3) or (tok.isalpha() and len(tok) < 3):
                continue
            out.append(tok)
    return out


class HistorySearch:
    def __init__(self, cassandra_repo, counter: Optional[TokenCounter] = None,
                 cfg: SystemSettings = default_settings) -> None:
        self.repo = cassandra_repo
        self.counter = counter or TokenCounter()
        self.cfg = cfg

    async def find(self, conversation_id: str, query: Optional[str], exclude_ids: Set[str],
                   older_than: Optional[Tuple[Any, str]] = None) -> List[Dict[str, Any]]:
        """Excerpts (oldest first) relevant to `query`, drawn from messages older than `older_than` (a message key) and
        not in `exclude_ids`. Returns [] when nothing qualifies."""
        if not self.cfg.HISTORY_SEARCH_ENABLED or not (query or "").strip():
            HISTORY_SEARCH.labels(outcome="skipped").inc()
            return []
        started = time.monotonic()
        try:
            qterms = list(dict.fromkeys(_terms(query)))
            recall = bool(RECALL_CUES.search(query))
            if len(qterms) < 2 and not recall:
                HISTORY_SEARCH.labels(outcome="skipped").inc()
                return []
            rows = await asyncio.to_thread(self.repo.get_recent_messages, conversation_id, self.cfg.HISTORY_SEARCH_MAX_MESSAGES)
            hits = self._rank(rows, qterms, query, recall, exclude_ids, older_than)
            result = self._excerpts(rows, hits, exclude_ids)
            HISTORY_SEARCH.labels(outcome="hit" if result else "miss").inc()
            return result
        except Exception:
            HISTORY_SEARCH.labels(outcome="error").inc()
            return []
        finally:
            HISTORY_SEARCH_SECONDS.observe(time.monotonic() - started)

    # ─── scoring ─────────────────────────────────────────────────────────────────────────────────────────────────────

    def _rank(self, rows, qterms, query, recall, exclude_ids, older_than) -> List[Dict[str, Any]]:
        candidates = [m for m in rows
                      if str(m.get("message_id")) not in exclude_ids
                      and len(str(m.get("content") or "").strip()) >= MIN_CONTENT_CHARS
                      and (older_than is None or message_key(m) < older_than)]
        if not candidates or not qterms:
            return []
        docs = [(m, Counter(_terms(str(m.get("content") or "")))) for m in candidates]
        n = len(docs)
        df = {t: sum(1 for _, tf in docs if t in tf) for t in qterms}
        idf = {t: math.log(1.0 + n / (1.0 + df[t])) for t in qterms}
        total = sum(idf.values()) or 1.0
        floor = self.cfg.HISTORY_SEARCH_RECALL_MIN_COVERAGE if recall else self.cfg.HISTORY_SEARCH_MIN_COVERAGE
        phrase = " ".join(_terms(query)[:6])
        scored = []
        for m, tf in docs:
            matched = [t for t in qterms if t in tf]
            if not matched or (len(qterms) >= 3 and len(matched) < 2):
                continue
            coverage = sum(idf[t] for t in matched) / total
            text = str(m.get("content") or "").lower()
            if len(qterms) >= 2 and phrase and phrase in " ".join(_terms(text)):
                coverage += 0.15                                        # the question's own wording appears
            if any(any(c.isdigit() for c in t) or any(c in t for c in "._-/:") for t in matched):
                coverage += 0.1                                         # identifier / code / error-like term matched
            if coverage >= floor:
                scored.append((min(1.0, coverage), message_key(m), m))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)           # best match first, newer wins ties
        return [m for _, _, m in scored[: self.cfg.HISTORY_SEARCH_TOP_K]]

    # ─── excerpts: a hit plus its question/answer partner, clipped, bounded in total ─────────────────────────────────

    def _excerpts(self, rows, hits, exclude_ids) -> List[Dict[str, Any]]:
        if not hits:
            return []
        ordered = sorted(rows, key=message_key)
        position = {str(m.get("message_id")): i for i, m in enumerate(ordered)}
        picked: Dict[str, Dict[str, Any]] = {}
        budget = self.cfg.HISTORY_TOTAL_MAX_TOKENS
        for hit in hits:
            i = position.get(str(hit.get("message_id")))
            if i is None:
                continue
            partner_index = i + 1 if str(hit.get("role")) == "user" else i - 1
            group = [hit]
            if 0 <= partner_index < len(ordered):
                partner = ordered[partner_index]
                if (str(partner.get("message_id")) not in exclude_ids and partner.get("role") != hit.get("role")):
                    group.append(partner)
            for m in sorted(group, key=message_key):
                mid = str(m.get("message_id"))
                if mid in picked:
                    continue
                text = self.counter.clip(str(m.get("content") or "").strip(), self.cfg.HISTORY_RESULT_MAX_TOKENS)
                cost = self.counter.count(text) + 4
                if cost > budget:
                    continue
                budget -= cost
                picked[mid] = {**m, "content": text}
        return sorted(picked.values(), key=message_key)
