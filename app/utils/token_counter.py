"""
app/utils/token_counter.py

Token counting for memory maintenance and validation. Memory Service had no tokenizer (it only used `len // 4`
estimates for metrics). Summaries are validated against a token limit and compaction pressure is measured in tokens, so
a real count is needed. It uses the same encoding and the same `len // 4` fallback as llm-service's TokenCounter
(tiktoken `cl100k_base`), so both services measure alike.
"""

import logging
from typing import Optional

logger = logging.getLogger("memory_service.utils.token_counter")


class TokenCounter:
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._enc = None
        try:
            import tiktoken
            self._enc = tiktoken.get_encoding(encoding_name)
        except Exception as exc:                       # missing package / offline first start: degrade, never fail
            logger.warning("tiktoken unavailable; using a 4-characters-per-token estimate (%s)", type(exc).__name__)

    @property
    def exact(self) -> bool:
        return self._enc is not None

    def count(self, text: Optional[str]) -> int:
        if not text:
            return 0
        if self._enc is not None:
            try:
                return max(1, len(self._enc.encode(text, disallowed_special=())))
            except Exception:
                pass
        return max(1, len(text) // 4)

    def clip(self, text: str, max_tokens: int) -> str:
        """Longest prefix that fits `max_tokens`, cut at a paragraph, line or sentence boundary when one exists in the
        last third of the prefix (so meaning is not chopped mid-sentence); an ellipsis marks a shortened text."""
        if max_tokens <= 0 or not text:
            return ""
        if self.count(text) <= max_tokens:
            return text
        marker = "…"
        budget = max(1, max_tokens - self.count(marker))
        if self._enc is not None:
            try:
                prefix = self._enc.decode(self._enc.encode(text, disallowed_special=())[:budget])
            except Exception:
                prefix = text[: budget * 4]
        else:
            prefix = text[: budget * 4]
        floor = int(len(prefix) * 2 / 3)
        for sep in ("\n\n", "\n", ". ", "? ", "! ", "; "):
            cut = prefix.rfind(sep)
            if cut >= floor:
                prefix = prefix[: cut + len(sep.rstrip(" "))]
                break
        prefix = prefix.rstrip()
        while prefix and self.count(prefix + marker) > max_tokens:      # tokenizer drift: shave until it fits
            prefix = prefix[: int(len(prefix) * 0.95)].rstrip()
        return (prefix + marker) if prefix else ""
