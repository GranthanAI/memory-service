"""
app/services/memory_policy.py

What deserves to be remembered, and how candidate facts relate to what is already stored.

The LLM proposes candidates; this module is the deterministic gate that decides what is stored,
so quality does not depend on the model following its prompt:

* Only durable, user-specific statements are kept: stable preferences, long-term goals, ongoing
  projects, durable personal facts, and interests with evidence of persistence. A question the user
  asked, a one-off task or a transient state ("is debugging this error today") is NOT a memory.
* Frequency is not importance. Interests are capped at INTEREST_CAP; only the same interest seen in
  a different conversation raises it (repetition across conversations is the evidence of persistence).
* Statements are normalized and compared without case, punctuation or filler, so rewording does not
  create a new memory.
* A statement that says the previous one changed ("now prefers", "instead", "no longer") replaces
  the closest existing memory instead of living next to it.
"""

import re
from dataclasses import dataclass
from typing import Optional

# Canonical categories stored in Cassandra/Milvus. (Existing data used "preferences", "habits",
# "plans", "goals": they map onto the canonical names on read/write.)
CATEGORIES = ("preferences", "goals", "projects", "personal", "interests")

_ALIASES = {
    "preference": "preferences", "preferences": "preferences", "habit": "preferences", "habits": "preferences",
    "style": "preferences", "communication": "preferences", "response_style": "preferences",
    "goal": "goals", "goals": "goals", "plan": "goals", "plans": "goals", "objective": "goals",
    "project": "projects", "projects": "projects", "work": "projects", "task": "projects",
    "personal": "personal", "fact": "personal", "facts": "personal", "general": "personal",
    "background": "personal", "profile": "personal", "skill": "personal", "skills": "personal",
    "role": "personal", "identity": "personal",
    "interest": "interests", "interests": "interests", "topic": "interests", "topics": "interests",
}

MIN_IMPORTANCE = 0.5
MAX_STATEMENT_CHARS = 300
MIN_STATEMENT_CHARS = 8
INTEREST_CAP = 0.6                  # a topic the user keeps asking about is not a durable preference
CORROBORATED_INTEREST_CAP = 0.75    # ...unless it recurs across conversations
CORROBORATION_BOOST = 0.1

# Default similarity (cosine) above which two statements are the same memory, per category.
SIMILARITY_THRESHOLDS = {"preferences": 0.85, "goals": 0.85, "projects": 0.85, "personal": 0.88, "interests": 0.75}
UPDATE_SIMILARITY = 0.6             # lower bar when the new statement says it replaces an older one

# Statements about the conversation or a transient state, not about the user.
_TEMPORARY = re.compile(
    r"\b(asked|asking|asks|wants? to know|wanted to know|is curious|was curious|question (about|on)|"
    r"wondering|inquir\w+|needs? help with|is trying to|tried to|right now|at the moment|"
    r"today|tonight|this (morning|afternoon|evening|week)|"
    r"currently (debugging|fixing|stuck|working on this)|"
    r"this (error|bug|issue|exception|traceback|snippet|code|file|question|query))\b",
    re.IGNORECASE,
)
_UPDATE = re.compile(
    r"\b(now|no longer|not anymore|anymore|instead|switched to|changed (to|their)|used to|"
    r"updated?|moved to|replaced|from now on|rather than)\b",
    re.IGNORECASE,
)
_EXPLICIT = re.compile(r"\b(remember|always|never|from now on|prefers?|wants? me to)\b", re.IGNORECASE)

_FILLER = re.compile(r"^(the )?user\s+(is\s+|has\s+|was\s+)?", re.IGNORECASE)
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")


def canonical_category(raw: str) -> str:
    """Maps any model-proposed category onto the canonical set (unknown -> personal)."""
    key = _SPACES.sub("_", (raw or "").strip().lower())
    return _ALIASES.get(key, "personal")


def normalize_statement(statement: str) -> str:
    """Comparison key: case, punctuation, spacing and a leading 'User ...' do not matter."""
    text = _FILLER.sub("", (statement or "").strip())
    return _SPACES.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def has_update_language(statement: str) -> bool:
    return bool(_UPDATE.search(statement or ""))


def threshold_for(category: str, statement: str) -> float:
    """Similarity at which `statement` is treated as the same memory as an existing one."""
    base = SIMILARITY_THRESHOLDS.get(category, 0.85)
    return min(base, UPDATE_SIMILARITY) if has_update_language(statement) else base


@dataclass(frozen=True)
class Candidate:
    accepted: bool
    category: str
    importance: float
    statement: str
    reason: str = ""


def evaluate(category: str, importance: float, statement: str) -> Candidate:
    """Accept, adjust or reject one candidate fact."""
    text = _SPACES.sub(" ", (statement or "").strip())
    cat = canonical_category(category)
    try:
        score = float(importance)
    except (TypeError, ValueError):
        score = 0.5
    score = min(1.0, max(0.0, score))

    def reject(reason: str) -> Candidate:
        return Candidate(False, cat, score, text, reason)

    if len(text) < MIN_STATEMENT_CHARS:
        return reject("too_short")
    if len(text) > MAX_STATEMENT_CHARS:
        text = text[:MAX_STATEMENT_CHARS].rstrip()
    if _TEMPORARY.search(text):
        return reject("temporary_or_conversation_specific")
    if cat == "interests":
        score = min(score, INTEREST_CAP)
    elif _EXPLICIT.search(text):
        score = min(1.0, score + 0.05)
    if score < MIN_IMPORTANCE:
        return reject("low_importance")
    return Candidate(True, cat, round(score, 3), text)


def corroborated_importance(category: str, old_importance: float, new_importance: float) -> float:
    """Importance of a memory seen again in a DIFFERENT conversation. Only this raises an interest above
    its cap; repetition inside one conversation never does."""
    best = max(float(old_importance), float(new_importance))
    boosted = best + CORROBORATION_BOOST
    cap = CORROBORATED_INTEREST_CAP if category == "interests" else 1.0
    return round(min(cap, boosted), 3)


def other_conversation(old_conversation_id: Optional[str], new_conversation_id: str) -> bool:
    return bool(old_conversation_id) and str(old_conversation_id) != str(new_conversation_id)
