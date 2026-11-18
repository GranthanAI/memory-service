"""
A shared Groq token budget, one per Groq MODEL (Groq limits tokens per minute per model, not per key).

llm-service (answers, request analysis) and memory-service (summaries, facts) share the same buckets through one Redis
(GROQ_BUDGET_REDIS_URL, keys `groq:budget:<model>:*`). The two services keep identical copies of this module.

  * After every Groq response the REAL numbers are stored: `x-ratelimit-remaining-tokens` and `x-ratelimit-reset-tokens`
    (key TTL = the reset time).
  * Before every call the caller asks `acquire(model, estimate, purpose)`: 0.0 = granted (the estimate is claimed), else the
    seconds until it would fit. Where the headers are not known yet, a sliding 60 s window of our own claims is checked
    against GROQ_TPM_LIMIT.
  * Priority: answers may use the whole bucket. Everything else (analysis, memory jobs) must leave GROQ_ANSWER_RESERVE_TOKENS
    for answers.
Redis trouble never blocks a call: it falls back to the in-process window.
"""

import re
import time
import uuid
from collections.abc import Callable
from typing import Any

from prometheus_client import Counter, Gauge

PREFIX = "groq:budget:"
ANSWER = "answer"
BUDGET_REMAINING = Gauge("groq_budget_remaining", "Groq tokens left in the current minute, from the last response headers", ["model"])
ANALYSIS_SKIPPED = Counter("groq_analysis_skipped_total", "Request analyses skipped so they do not eat the answer reserve")
BUDGET_WAITS = Counter("groq_budget_waits_total", "Calls that found a Groq model's budget short", ["purpose"])

_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h)")


def parse_reset_seconds(value: Any) -> float | None:
    """Groq's reset header ("7.66s", "1m30.5s", "120ms" or a bare number) in seconds; None when it cannot be read."""
    if value is None:
        return None
    text = str(value).strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    parts = _DURATION.findall(text)
    if not parts:
        return None
    unit = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    return sum(float(n) * unit[u] for n, u in parts)


def read_headers(headers: Any) -> tuple[int, float] | None:
    """(remaining tokens, reset seconds) from a response's headers, else None."""
    try:
        remaining = headers.get("x-ratelimit-remaining-tokens")
        reset = parse_reset_seconds(headers.get("x-ratelimit-reset-tokens"))
        if remaining is None or reset is None:
            return None
        return int(float(str(remaining))), reset
    except (AttributeError, TypeError, ValueError):
        return None


class GroqBudget:
    def __init__(self, redis: Any = None, *, tpm_limit: int = 8000, answer_reserve: int = 4000,
                 clock: Callable[[], float] = time.time) -> None:
        self.redis = redis
        self.tpm_limit = tpm_limit
        self.answer_reserve = answer_reserve
        self._now = clock
        self._window: dict[str, list[tuple[float, int]]] = {}  # in-process copy when Redis is unavailable

    def _floor(self, purpose: str) -> int:
        return 0 if purpose == ANSWER else self.answer_reserve

    # --- from the provider -------------------------------------------------------------------------------------------
    async def record(self, model: str, remaining: int, reset_s: float) -> None:
        BUDGET_REMAINING.labels(model=model).set(remaining)
        if self.redis is None:
            return
        try:
            await self.redis.set(f"{PREFIX}{model}:remaining", max(0, remaining), ex=max(1, int(reset_s + 0.999)))
        except Exception:  # noqa: BLE001
            pass

    async def record_headers(self, model: str, headers: Any) -> bool:
        got = read_headers(headers)
        if got is None:
            return False
        await self.record(model, got[0], got[1])
        return True

    async def record_error(self, model: str, exc: Exception) -> None:
        """A 429 carries the same headers: after it the bucket is known to be (nearly) empty."""
        headers = getattr(getattr(exc, "response", None), "headers", None)
        if headers is not None:
            await self.record_headers(model, headers)

    async def settle(self, model: str, estimate: int, actual: int) -> None:
        """Give back (or take) the difference between what was claimed and what the call really used (no headers case)."""
        if self.redis is None or estimate == actual:
            return
        try:
            if await self.redis.exists(f"{PREFIX}{model}:remaining"):
                await self.redis.incrby(f"{PREFIX}{model}:remaining", estimate - actual)
        except Exception:  # noqa: BLE001
            pass

    # --- before a call -----------------------------------------------------------------------------------------------
    async def known_remaining(self, model: str) -> tuple[int, float] | None:
        """(tokens left, seconds until reset) while the last response's headers are still current."""
        if self.redis is None:
            return None
        try:
            key = f"{PREFIX}{model}:remaining"
            raw = await self.redis.get(key)
            if raw is None:
                return None
            ttl = await self.redis.ttl(key)
            return int(raw), float(ttl if ttl and ttl > 0 else 1)
        except Exception:  # noqa: BLE001
            return None

    async def acquire(self, model: str, estimate: int, purpose: str = ANSWER) -> float:
        """0.0 = granted and claimed; else the seconds until `estimate` tokens would fit for this purpose."""
        floor = self._floor(purpose)
        known = await self.known_remaining(model)
        if known is not None:
            left, reset = known
            if estimate <= left - floor:
                try:
                    await self.redis.decrby(f"{PREFIX}{model}:remaining", estimate)
                    BUDGET_REMAINING.labels(model=model).set(left - estimate)
                except Exception:  # noqa: BLE001
                    pass
                return 0.0
            BUDGET_WAITS.labels(purpose=purpose).inc()
            return max(reset, 0.1)
        wait = await self._window_acquire(model, estimate, floor)
        if wait > 0:
            BUDGET_WAITS.labels(purpose=purpose).inc()
        return wait

    async def _window_acquire(self, model: str, estimate: int, floor: int) -> float:
        if self.tpm_limit <= 0:
            return 0.0  # no ceiling configured
        now = self._now()
        limit = self.tpm_limit - floor
        entries = await self._load_window(model, now)
        used = sum(t for _, t in entries)
        if used + estimate <= limit or not entries:
            await self._store_claim(model, now, estimate)
            return 0.0
        # wait until enough of the oldest claims age out of the 60 s window
        need = used + estimate - limit
        freed = 0
        for ts, tokens in sorted(entries):
            freed += tokens
            if freed >= need:
                return max(ts + 60.0 - now, 0.1)
        return 60.0

    async def _load_window(self, model: str, now: float) -> list[tuple[float, int]]:
        if self.redis is not None:
            try:
                key = f"{PREFIX}{model}:win"
                await self.redis.zremrangebyscore(key, 0, now - 60.0)
                members = await self.redis.zrange(key, 0, -1, withscores=True)
                out = []
                for member, score in members:
                    text = member.decode() if isinstance(member, bytes) else member
                    out.append((float(score), int(text.split(":")[1])))
                return out
            except Exception:  # noqa: BLE001
                pass
        window = [(t, n) for t, n in self._window.get(model, []) if t > now - 60.0]
        self._window[model] = window
        return list(window)

    async def _store_claim(self, model: str, now: float, tokens: int) -> None:
        if self.redis is not None:
            try:
                key = f"{PREFIX}{model}:win"
                await self.redis.zadd(key, {f"{uuid.uuid4().hex[:8]}:{tokens}": now})
                await self.redis.expire(key, 90)
                return
            except Exception:  # noqa: BLE001
                pass
        self._window.setdefault(model, []).append((now, tokens))


def budget_from_settings(redis_url: str, tpm_limit: int, answer_reserve: int) -> GroqBudget:
    """A budget on the shared Redis; with no Redis reachable it still works from its in-process window."""
    redis = None
    try:
        import redis.asyncio as aioredis

        redis = aioredis.from_url(redis_url, decode_responses=False, socket_timeout=0.5, socket_connect_timeout=0.5) if redis_url else None
    except Exception:  # noqa: BLE001
        redis = None
    return GroqBudget(redis, tpm_limit=tpm_limit, answer_reserve=answer_reserve)


class GroqBudgetDeferred(Exception):
    """A memory job found the Groq budget short: re-queue it after `wait_s` seconds (this is not a failure, it uses no attempt)."""

    def __init__(self, wait_s: float, model: str = "") -> None:
        super().__init__(f"Groq token budget for {model or 'the model'} is short for {wait_s:.0f}s; deferred")
        self.wait_s = wait_s


MEMORY_JOBS_DEFERRED = Counter("memory_jobs_deferred_total", "Memory jobs re-queued because the shared Groq budget was short", ["job_type"])

_BUDGET: "GroqBudget | None" = None


def get_budget() -> GroqBudget:
    """The process-wide budget, built from settings on first use."""
    global _BUDGET
    if _BUDGET is None:
        from app.core.config import settings

        _BUDGET = budget_from_settings(settings.GROQ_BUDGET_REDIS_URL, settings.GROQ_TPM_LIMIT, settings.GROQ_ANSWER_RESERVE_TOKENS)
    return _BUDGET
