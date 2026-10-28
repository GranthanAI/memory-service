"""Synthetic long conversations with PLANTED facts whose survival can be measured exactly.

Each planted fact has a unique, searchable token. Kinds:
  identifier  an id / port / path stated once, never repeated
  constraint  a standing requirement from the user
  decision    a decision that is LATER REVERSED (the final option must win; the stale one is history)
  open_item   a question that is either resolved later ("done") or stays open
"""
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
FILLER = ["Thanks, that makes sense.", "Can you explain that a bit more?", "Interesting, let's keep going.",
          "I see. What are the trade-offs here?", "Okay, noted.", "Could you give a short example?",
          "That matches what I expected.", "Let's move on to the next part."]


def v7(ts: datetime, rand: int) -> str:
    return str(uuid.UUID(int=(int(ts.timestamp() * 1000) << 80) | (0x7 << 76) | (0b10 << 62) | rand))


@dataclass
class Planted:
    kind: str                 # identifier | constraint | decision | open_item
    token: str                # what must be findable in the final context
    index: int                # message index it was planted at
    stale_token: str = ""     # for reversed decisions: the superseded option
    resolved: bool = False    # for open items: closed by a later message
    reversed: bool = False    # for decisions: a later message really switched to `token`


@dataclass
class Conversation:
    messages: List[Dict]
    planted: List[Planted] = field(default_factory=list)


def build_conversation(seed: int, length: int = 120) -> Conversation:
    """`length` messages (alternating user / assistant) with facts planted at seeded positions."""
    rng = random.Random(seed)
    slots = sorted(rng.sample(range(0, length, 2), k=min(24, length // 4)))          # user-message positions
    plan: Dict[int, Planted] = {}
    texts: Dict[int, str] = {}
    counters = {"id": 0, "con": 0, "dec": 0, "open": 0}

    def token(prefix: str, key: str) -> str:
        counters[key] += 1
        return f"{prefix}{seed % 97:02d}x{counters[key]:02d}"

    reversals: List[Planted] = []
    for position, slot in enumerate(slots):
        kind = ("identifier", "constraint", "decision", "open_item")[position % 4]
        if kind == "identifier":
            t = token("srv", "id")
            plan[slot] = Planted("identifier", t, slot)
            texts[slot] = f"For the record, the staging service id is {t} and it listens on port {8000 + rng.randrange(900)}."
        elif kind == "constraint":
            t = token("rule", "con")
            plan[slot] = Planted("constraint", t, slot)
            texts[slot] = f"Requirement {t}: always answer with a short summary first."
        elif kind == "decision":
            old, new = token("optA", "dec"), token("optB", "dec")
            plan[slot] = Planted("decision", new, slot, stale_token=old)
            texts[slot] = f"We decided to use {old} for the queue because it is simple."
            reversals.append(plan[slot])
        else:
            t = token("task", "open")
            resolved = position % 8 == 3
            plan[slot] = Planted("open_item", t, slot, resolved=resolved)
            texts[slot] = f"TODO: someone must still decide {t}."
    # reversals and resolutions come later in the conversation
    for planted in reversals:
        later = [i for i in range(planted.index + 20, length, 2) if i not in texts]
        if later:
            slot = later[rng.randrange(len(later))]
            texts[slot] = f"Correction: we no longer use {planted.stale_token}; we switched to {planted.token} instead."
            planted.reversed = True
    for planted in [p for p in plan.values() if p.kind == "open_item" and p.resolved]:
        later = [i for i in range(planted.index + 10, length, 2) if i not in texts]
        if later:
            texts[later[rng.randrange(len(later))]] = f"DONE: {planted.token} is decided and finished."

    for planted in reversals:
        if not planted.reversed:                      # never reversed: the original option is the current one
            planted.token, planted.stale_token = planted.stale_token, ""
    messages: List[Dict] = []
    for i in range(length):
        ts = T0 + timedelta(seconds=30 * i)
        if i % 2 == 0:
            content = texts.get(i, FILLER[rng.randrange(len(FILLER))])
            role = "user"
        else:
            content = "Understood. " + FILLER[rng.randrange(len(FILLER))]
            role = "assistant"
        messages.append({"message_id": v7(ts, i + 1), "role": role, "content": content, "created_at": ts})
    return Conversation(messages, sorted(plan.values(), key=lambda p: p.index))
