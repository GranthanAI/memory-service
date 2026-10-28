"""
app/services/anchors.py

The anchor ledger: facts of a conversation that must never be lost to summarization - identifiers (ids, numbers, paths,
versions, URLs), decisions (with their reason), constraints, open items and corrections - each with the id of the message
it came from. Anchors are stored outside the summary text (conversation_anchors), are never trimmed, and are rendered next to
the summary; the summary can therefore be rewritten and shortened without a model quietly dropping "port 8443".

Extraction is two-layered: deterministic patterns for identifiers (cheap, exact, no model), plus an LLM pass for the
judgement-based kinds (decision / constraint / open_item / correction / done). Validation helpers here are deterministic.
"""
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

KINDS = ("identifier", "decision", "constraint", "open_item", "correction")
JUDGED_KINDS = ("decision", "constraint", "open_item", "correction")
MAX_ANCHOR_CHARS = 300

# Hard entities: things that are either right or wrong, so a summary must not invent or alter them.
_PATTERNS = [
    re.compile(r"https?://[^\s<>\")']+"),                                                # URLs
    re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),                                     # emails
    re.compile(r"(?<![\w/])(?:[A-Za-z]:\\|\.{0,2}/)[\w.\-/\\]*[\w.\-]{2,}"),             # file system paths
    re.compile(r"\b[\w\-]+(?:/[\w.\-]+)+\.[A-Za-z0-9]{1,6}\b"),                          # relative file paths with extension
    re.compile(r"\b[\w\-]+\.(?:py|js|ts|tsx|java|go|rs|rb|php|cs|cpp|c|h|sql|ya?ml|json|toml|md|txt|sh|cfg|ini|env|html|css)\b"),
    re.compile(r"\bv?\d+\.\d+(?:\.\d+){0,2}(?:[-+][\w.]+)?\b"),                          # versions / decimals
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I),  # uuids
    re.compile(r"\b[0-9a-f]{7,40}\b"),                                                   # commit hashes
    re.compile(r"\b[A-Za-z]+[_-][A-Za-z0-9_-]*\d[\w-]*\b|\b[A-Za-z]*\d+[A-Za-z]+\d*\b"),  # ids like req_42, abc123, x2y
    # ENV_VARS / CONSTANTS: must contain "_" or a digit; plain CAPS words ("CURRENT STATE", "KEY FACTS") are headings
    re.compile(r"\b[A-Z][A-Z0-9]*[_0-9][A-Z0-9_]*\b"),
    re.compile(r"\b[a-z]+(?:[A-Z][a-z0-9]+)+\b|\b(?:[A-Z][a-z0-9]+){2,}\b"),             # camelCase / PascalCase identifiers
    re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?%?(?!\w|\.\d)"),                           # numbers, money, percentages
]
# plain English that matches the patterns above but is not an entity
_STOP = {"this", "that", "with", "from", "have", "will", "your", "what", "when", "where", "which", "there", "their",
         "about", "would", "could", "should", "these", "those", "into", "than", "then", "them", "they", "been", "were",
         "also", "only", "some", "such", "very", "just", "like", "make", "made", "more", "most", "much", "many", "need",
         "user", "assistant", "http", "https", "true", "false", "none", "null", "json", "note", "todo", "note:", "open"}
_PURE_SMALL_NUMBER = re.compile(r"^\$?[0-9]{1,2}$")       # "1", "12": too common to be an identifier worth pinning


def mentions(haystack_norm: str, needle: str, threshold: float = 0.7) -> bool:
    """True when most of the needle's meaningful words appear in the haystack (a paraphrase still counts)."""
    needle_norm = normalize(needle)
    if not needle_norm:
        return True
    if needle_norm in haystack_norm:
        return True
    words = [w for w in re.findall(r"\w+", needle_norm) if len(w) > 3]
    if not words:
        return False
    return sum(1 for w in words if w in haystack_norm) / len(words) >= threshold


@dataclass(frozen=True)
class Anchor:
    kind: str
    text: str
    source_message_id: str = ""
    status: str = "active"            # active | open | done | superseded

    @property
    def anchor_id(self) -> str:
        return hashlib.sha1(f"{self.kind}|{normalize(self.text)}".encode("utf-8")).hexdigest()[:20]


def normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def hard_entities(text: str) -> List[str]:
    """Identifier-like tokens in `text` (deduplicated, in order of appearance)."""
    found: List[tuple] = []
    for pattern in _PATTERNS:
        for m in pattern.finditer(text or ""):
            token = m.group(0).strip(".,;:()[]{}\"'")
            if len(token) < 2 or token.lower() in _STOP or _PURE_SMALL_NUMBER.match(token):
                continue
            found.append((m.start(), token))
    seen, out = set(), []
    for _pos, token in sorted(found):
        key = token.lower()
        if key not in seen:
            seen.add(key)
            out.append(token)
    return out


def extract_identifiers(messages: Sequence[Dict[str, Any]], limit: int = 40) -> List[Anchor]:
    """Deterministic identifier anchors from message text, each tied to the message it first appeared in."""
    anchors: List[Anchor] = []
    seen = set()
    for m in messages:
        for token in hard_entities(str(m.get("content") or "")):
            key = token.lower()
            if key in seen:
                continue
            seen.add(key)
            anchors.append(Anchor("identifier", token, str(m.get("message_id") or "")))
            if len(anchors) >= limit:
                return anchors
    return anchors


def parse_llm_anchors(raw: str, messages: Sequence[Dict[str, Any]]) -> tuple:
    """(anchors, done_texts) from the model's JSON answer. Raises ValueError on an unusable answer.

    Expected: {"anchors": [{"kind": "...", "text": "...", "message": <0-based index>}...], "done": ["<open item text>", ...]}"""
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        raise ValueError("anchor extraction returned no JSON")
    data = json.loads(match.group(0))
    anchors: List[Anchor] = []
    for item in data.get("anchors") or []:
        kind, text = str(item.get("kind") or "").strip().lower(), " ".join(str(item.get("text") or "").split())
        if kind not in JUDGED_KINDS or not text:
            continue
        index = item.get("message")
        source = str(messages[index].get("message_id")) if isinstance(index, int) and 0 <= index < len(messages) else ""
        anchors.append(Anchor(kind, text[:MAX_ANCHOR_CHARS], source, "open" if kind == "open_item" else "active"))
    done = [" ".join(str(t).split()) for t in (data.get("done") or []) if str(t).strip()]
    return anchors, done


def merge_ledger(existing: Iterable[Dict[str, Any]], new: Iterable[Anchor], done_texts: Sequence[str]) -> List[Anchor]:
    """The ledger after this batch: existing anchors kept (nothing is dropped), new ones added, and open items that a new
    message marked as finished flipped to done. Matching a 'done' mark to an open item is by normalized text containment."""
    by_id: Dict[str, Anchor] = {}
    for row in existing:
        a = Anchor(row["kind"], row["text"], row.get("source_message_id") or "", row.get("status") or "active")
        by_id[a.anchor_id] = a
    for a in new:
        by_id.setdefault(a.anchor_id, a)
    for done in done_texts:
        nd = normalize(done)
        for aid, a in list(by_id.items()):
            if a.kind == "open_item" and a.status == "open" and nd and (mentions(normalize(a.text), done, 0.75) or mentions(nd, a.text, 0.75)):
                by_id[aid] = Anchor(a.kind, a.text, a.source_message_id, "done")
    return list(by_id.values())


# ─── validation of a freshly written summary ────────────────────────────────────────────────────────────────────────────


@dataclass
class ValidationReport:
    lost_anchors: List[str] = field(default_factory=list)          # existed before, now in neither the summary nor the ledger
    dropped_open_items: List[str] = field(default_factory=list)    # open items removed without a message marking them done
    unsupported_entities: List[str] = field(default_factory=list)  # in the new summary, in neither old summary nor new messages

    @property
    def ok(self) -> bool:
        return not (self.lost_anchors or self.dropped_open_items or self.unsupported_entities)

    def feedback(self) -> str:
        lines: List[str] = []
        if self.lost_anchors:
            lines.append("These must appear in the summary again: " + "; ".join(self.lost_anchors))
        if self.dropped_open_items:
            lines.append("These OPEN ITEMS were removed but nothing says they are finished; keep them: "
                         + "; ".join(self.dropped_open_items))
        if self.unsupported_entities:
            lines.append("These names/numbers/identifiers are not in the previous summary or the new messages; remove or "
                         "correct them: " + "; ".join(self.unsupported_entities))
        return "\n".join(lines)


def validate_summary(
    *, new_summary: str, previous_summary: str, new_messages_text: str,
    prior_ledger: Sequence[Dict[str, Any]], merged_ledger: Sequence[Anchor],
) -> ValidationReport:
    """Deterministic checks (the token cap is enforced separately by the fitting step):
      * every anchor that existed before is in the summary or the ledger;
      * no hard entity appears that is in neither the previous summary, the new messages nor the ledger;
      * an open item leaves the summary only when it is marked done in the ledger."""
    report = ValidationReport()
    summary_norm = normalize(new_summary)
    ledger_norm = {normalize(a.text) for a in merged_ledger}

    # 1. nothing that was pinned before has disappeared
    for row in prior_ledger:
        text = normalize(row["text"])
        if text not in ledger_norm and text not in summary_norm:
            report.lost_anchors.append(row["text"])
    for token in hard_entities(previous_summary):
        if normalize(token) not in summary_norm and normalize(token) not in " | ".join(ledger_norm):
            report.lost_anchors.append(token)

    # 2. no invented entities
    allowed = normalize(" ".join([previous_summary, new_messages_text, " ".join(a.text for a in merged_ledger)]))
    for token in hard_entities(new_summary):
        if normalize(token) not in allowed:
            report.unsupported_entities.append(token)

    # 3. open items stay until marked done
    previous_norm = normalize(previous_summary)
    for a in merged_ledger:
        if a.kind == "open_item" and a.status == "open":
            if mentions(previous_norm, a.text) and not mentions(summary_norm, a.text):
                report.dropped_open_items.append(a.text)
    return report


# ─── rendering ──────────────────────────────────────────────────────────────────────────────────────────────────────────


def render_order(anchors: Iterable[Dict[str, Any]], max_chars: int) -> List[Dict[str, Any]]:
    """Anchors to show next to the summary: everything still true, open items first, then constraints, decisions,
    corrections, identifiers; closed (done/superseded) ones are omitted. If the ledger is huge the newest are dropped from
    the identifier tail first, never a constraint, decision, correction or open item."""
    rank = {"open_item": 0, "constraint": 1, "decision": 2, "correction": 3, "identifier": 4}
    live = [a for a in anchors if (a.get("status") or "active") in ("active", "open")]
    live.sort(key=lambda a: (rank.get(a["kind"], 9), str(a.get("created_at") or "")))
    out, used = [], 0
    for a in live:
        size = len(a["text"]) + 14
        if used + size > max_chars and a["kind"] == "identifier":
            continue
        used += size
        out.append({"kind": a["kind"], "text": a["text"], "source_message_id": a.get("source_message_id") or ""})
    return out
