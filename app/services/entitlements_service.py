"""Per-user entitlements (plan limits) cached in Redis (this service has Redis; same keys and fallback as conversation-service).

  ent:{user_id}         the last GetEntitlements answer, TTL ENTITLEMENTS_CACHE_SECONDS (60 s): shared by all replicas
  ent:stale:{user_id}   the same value kept ENTITLEMENTS_STALE_SECONDS (1 h): used only when billing is unreachable
  nothing cached + billing unreachable -> the bundled Free limits (source "free_fallback", app/core/plan_defaults.py)

-1 means unlimited. `entitlements.changed` and `user.deleted` events call `invalidate`.
"""

import json
import logging
from typing import Optional

from app.clients.billing_client import BillingClient, BillingUnavailable, EntitlementsData
from app.core.config import settings
from app.core.plan_defaults import FREE_LIMITS

logger = logging.getLogger("memory_service.services.entitlements_service")


def cache_key(user_id) -> str:
    return f"ent:{user_id}"


def stale_key(user_id) -> str:
    return f"ent:stale:{user_id}"


def free_entitlements(user_id) -> EntitlementsData:
    return EntitlementsData(user_id=str(user_id), plan_code="free", limits=dict(FREE_LIMITS), source="free_fallback")


def limit_of(ent: EntitlementsData, name: str) -> int:
    """The limit's value; -1 = unlimited. A limit missing from the answer falls back to the Free value."""
    value = ent.limits.get(name)
    return int(value) if value is not None else FREE_LIMITS.get(name, 0)


class EntitlementsService:
    def __init__(self, billing: BillingClient, redis_client=None):
        self.billing = billing
        self.redis = redis_client

    async def get(self, user_id) -> EntitlementsData:
        uid = str(user_id)
        cached = await self._read(cache_key(uid))
        if cached is not None:
            return cached
        try:
            ent = await self.billing.get_entitlements(uid)
        except BillingUnavailable as e:
            stale = await self._read(stale_key(uid))
            if stale is not None:
                logger.warning("entitlements_stale_used reason=%s", e.reason)
                stale.source = "stale"
                return stale
            logger.warning("entitlements_free_fallback reason=%s", e.reason)
            return free_entitlements(uid)
        await self._write(uid, ent)
        return ent

    async def invalidate(self, user_id) -> None:
        """Drops both keys (entitlements.changed, user.deleted)."""
        if self.redis is None:
            return
        try:
            await self.redis.delete(cache_key(user_id), stale_key(user_id))
        except Exception as e:  # noqa: BLE001
            logger.warning("entitlements_invalidate_failed: %s", e)

    async def _read(self, key: str) -> Optional[EntitlementsData]:
        if self.redis is None:
            return None
        try:
            raw = await self.redis.get(key)
            return None if raw is None else EntitlementsData.from_json(json.loads(raw))
        except Exception as e:  # noqa: BLE001
            logger.warning("entitlements_cache_read_failed: %s", e)
            return None

    async def _write(self, uid: str, ent: EntitlementsData) -> None:
        if self.redis is None:
            return
        try:
            raw = json.dumps(ent.to_json())
            await self.redis.set(cache_key(uid), raw, ex=settings.ENTITLEMENTS_CACHE_SECONDS)
            await self.redis.set(stale_key(uid), raw, ex=settings.ENTITLEMENTS_STALE_SECONDS)
        except Exception as e:  # noqa: BLE001
            logger.warning("entitlements_cache_write_failed: %s", e)
