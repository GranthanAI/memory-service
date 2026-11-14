"""Async gRPC client for billing-service `BillingInternal`, GetEntitlements ONLY (integration plan 4.5).

memory-service reserves nothing (its model calls are cost-only, see usage_metering.py); it only reads the user's plan limits
(`memory_facts`). Same rules as conversation-service's and file-service's clients: deadline 300 ms, a circuit breaker that opens
after 5 consecutive failures or timeouts and goes half-open after 30 s, metadata `authorization: Bearer <token>`.

NOTE: billing's gRPC is BILLING_GRPC_TARGET (127.0.0.1:50061). This service's own gRPC server is on MEMORY_CONTEXT_GRPC_PORT (50051).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import grpc

from app.clients.billing_grpc import billing_pb2 as pb
from app.clients.billing_grpc import billing_pb2_grpc as pb_grpc
from app.core.config import settings

logger = logging.getLogger("memory_service.clients.billing_client")

_TRANSPORT_FAILURES = {
    grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED, grpc.StatusCode.INTERNAL,
    grpc.StatusCode.UNKNOWN, grpc.StatusCode.RESOURCE_EXHAUSTED, grpc.StatusCode.ABORTED,
}


class BillingUnavailable(Exception):
    """Billing could not answer (down, too slow, breaker open, or it refused the call itself)."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass
class EntitlementsData:
    user_id: str
    plan_code: str
    limits: Dict[str, int] = field(default_factory=dict)
    allowed_models: List[str] = field(default_factory=list)
    source: str = ""
    version_hash: str = ""

    def to_json(self) -> dict:
        return {"user_id": self.user_id, "plan_code": self.plan_code, "limits": self.limits,
                "allowed_models": self.allowed_models, "source": self.source, "version_hash": self.version_hash}

    @classmethod
    def from_json(cls, d: dict) -> "EntitlementsData":
        return cls(user_id=d.get("user_id", ""), plan_code=d.get("plan_code", "free"),
                   limits={k: int(v) for k, v in (d.get("limits") or {}).items()},
                   allowed_models=list(d.get("allowed_models") or []), source=d.get("source", ""),
                   version_hash=d.get("version_hash", ""))


class CircuitBreaker:
    """closed -> (N consecutive failures) -> open -> (reset_seconds) -> half-open (one probe) -> closed | open."""

    def __init__(self, failures: int, reset_seconds: float, clock: Callable[[], float] = time.monotonic):
        self.threshold, self.reset_seconds, self.clock = failures, reset_seconds, clock
        self.consecutive = 0
        self.opened_at: Optional[float] = None
        self.probing = False

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "half_open" if self.clock() - self.opened_at >= self.reset_seconds else "open"

    def allow(self) -> bool:
        state = self.state
        if state == "closed":
            return True
        if state == "half_open" and not self.probing:
            self.probing = True
            return True
        return False

    def success(self) -> None:
        self.consecutive, self.opened_at, self.probing = 0, None, False

    def failure(self) -> None:
        self.consecutive += 1
        if self.probing or self.consecutive >= self.threshold:
            self.opened_at = self.clock()
            self.probing = False
            logger.warning("billing_breaker_open consecutive_failures=%s", self.consecutive)


class BillingClient:
    def __init__(self, target: Optional[str] = None, token: Optional[str] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.target = target or settings.BILLING_GRPC_TARGET
        self.token = settings.BILLING_SERVICE_TOKEN if token is None else token
        self.breaker = CircuitBreaker(settings.BILLING_BREAKER_FAILURES, settings.BILLING_BREAKER_RESET_SECONDS, clock)
        self._channel = None
        self._stub = None

    def _get_stub(self):
        if self._stub is None:
            self._channel = grpc.aio.insecure_channel(self.target)
            self._stub = pb_grpc.BillingInternalStub(self._channel)
        return self._stub

    async def close(self) -> None:
        if self._channel is not None:
            await self._channel.close()
            self._channel, self._stub = None, None

    async def get_entitlements(self, user_id: str) -> EntitlementsData:
        if not self.breaker.allow():
            raise BillingUnavailable("breaker_open")
        try:
            r = await self._get_stub().GetEntitlements(
                pb.GetEntitlementsRequest(user_id=user_id), timeout=settings.BILLING_ENTITLEMENTS_DEADLINE_SECONDS,
                metadata=[("authorization", f"Bearer {self.token}")])
        except grpc.aio.AioRpcError as e:
            code = e.code()
            if code in _TRANSPORT_FAILURES:
                self.breaker.failure()
            else:                                     # billing answered but refused (bad token): a configuration problem
                self.breaker.success()
                logger.error("billing_call_refused code=%s detail=%s", code.name, e.details())
            raise BillingUnavailable(code.name.lower()) from e
        except Exception as e:  # noqa: BLE001
            self.breaker.failure()
            raise BillingUnavailable(type(e).__name__) from e
        self.breaker.success()
        return EntitlementsData(user_id=r.user_id or user_id, plan_code=r.plan_code or "free",
                                limits={k: int(v) for k, v in r.limits.items()}, allowed_models=list(r.allowed_models),
                                source=r.source, version_hash=r.version_hash)


billing_client = BillingClient()
