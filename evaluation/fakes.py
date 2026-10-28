"""Deterministic stand-ins for the model and the stores: no network, no paid calls.

`LossySummarizer` behaves like a real summarizer's WORST habits on purpose: it compresses by dropping the oldest bullets,
forgets identifiers, and now and then invents a number. `RuleExtractor` plays the anchor-extraction model (an oracle that
reads the planted markers). The point of the harness is to show what the ledger and the validation save."""
import random
import re
from typing import Dict, List, Optional, Tuple

from app.schemas.llm import SummarizeResponse
from app.services.anchors import Anchor

HEADINGS = ["CONSTRAINTS & PREFERENCES", "DECISIONS", "CORRECTIONS", "OPEN ITEMS", "KEY FACTS & OUTCOMES"]
_SECTION_OF = {"constraint": 0, "decision": 1, "correction": 2, "open_item": 3, "fact": 4}


def _parse(summary: str) -> List[Tuple[int, str]]:
    bullets, section = [], -1
    for line in (summary or "").split("\n"):
        line = line.rstrip()
        if line in HEADINGS:
            section = HEADINGS.index(line)
        elif line.startswith("- ") and section >= 0:
            bullets.append((section, line[2:]))
    return bullets


def _render(bullets: List[Tuple[int, str]]) -> str:
    out = []
    for i, heading in enumerate(HEADINGS):
        lines = [f"- {text}" for sec, text in bullets if sec == i]
        if lines:
            out += [heading, *lines]
    return "\n".join(out)


def _classify(content: str):
    """(kind, details) of a planted-marker message, else None."""
    if m := re.search(r"Requirement (\w+): (.*)", content):
        return "constraint", {"token": m.group(1), "text": f"Requirement {m.group(1)}: {m.group(2)}"}
    if m := re.search(r"Correction: we no longer use (\w+); we switched to (\w+)", content):
        return "correction", {"old": m.group(1), "new": m.group(2)}
    if m := re.search(r"We decided to use (\w+) for the queue because (.*)", content):
        return "decision", {"token": m.group(1), "text": f"We decided to use {m.group(1)} for the queue because {m.group(2)}"}
    if m := re.search(r"TODO: someone must still decide (\w+)", content):
        return "open_item", {"token": m.group(1), "text": f"Someone must still decide {m.group(1)}"}
    if m := re.search(r"DONE: (\w+) is decided", content):
        return "done", {"token": m.group(1)}
    if m := re.search(r"staging service id is (\w+) and it listens on port (\d+)", content):
        return "fact", {"token": m.group(1), "text": f"Staging service {m.group(1)} on port {m.group(2)}"}
    return None


class LossySummarizer:
    def __init__(self, seed: int, drop_fact_p: float = 0.7, max_bullets: int = 10, hallucinate_p: float = 0.25):
        self.rng = random.Random(seed)
        self.drop_fact_p, self.max_bullets, self.hallucinate_p = drop_fact_p, max_bullets, hallucinate_p
        self.calls = self.retries = 0

    async def summarize(self, request) -> SummarizeResponse:
        self.calls += 1
        bullets = _parse(request.previous_summary or "")
        for message in request.new_messages:
            parsed = _classify(message.content)
            if parsed is None:
                continue
            kind, d = parsed
            if kind == "correction":
                bullets = [(s, t) for s, t in bullets if not (s == 1 and d["old"] in t)]
                bullets.append((2, f"Was {d['old']}, now {d['new']}"))
                bullets.append((1, f"Use {d['new']} for the queue"))
            elif kind == "done":
                bullets = [(s, t) for s, t in bullets if not (s == 3 and d["token"] in t)]
            elif kind == "fact":
                if self.rng.random() >= self.drop_fact_p:                  # a real model forgets identifiers
                    bullets.append((4, d["text"]))
            else:
                bullets.append((_SECTION_OF[kind], d["text"]))
        if len(bullets) > self.max_bullets:                                # "compression": the oldest bullets go first
            bullets = bullets[len(bullets) - self.max_bullets:]
        if request.feedback:
            self.retries += 1
        elif self.rng.random() < self.hallucinate_p:
            bullets.append((4, "Fallback host listens on port 9999"))      # invented
        return SummarizeResponse(summary=_render(bullets) or "No notable content yet.")


class RuleExtractor:
    """Anchor extraction as a model would do it (an oracle over the planted markers)."""

    async def __call__(self, messages: List[Dict], open_items: List[str]) -> Tuple[List[Anchor], List[str]]:
        anchors, done = [], []
        for m in messages:
            parsed = _classify(str(m.get("content") or ""))
            if parsed is None:
                continue
            kind, d = parsed
            mid = str(m.get("message_id") or "")
            if kind in ("constraint", "decision", "open_item"):
                anchors.append(Anchor(kind, d["text"], mid, "open" if kind == "open_item" else "active"))
            elif kind == "correction":
                anchors.append(Anchor("correction", f"Was {d['old']}, now {d['new']}", mid, "active"))
            elif kind == "done":
                done.append(f"Someone must still decide {d['token']}")         # same wording as the open item it closes
        return anchors, done


class SummarizerWithExtractor:
    """The object SummaryService talks to: summarize() from the lossy model, extract_anchors() from the oracle."""

    def __init__(self, summarizer: LossySummarizer, extractor: Optional[RuleExtractor]):
        self.summarizer = summarizer
        if extractor is not None:
            self.extract_anchors = extractor

    async def summarize(self, request):
        return await self.summarizer.summarize(request)
