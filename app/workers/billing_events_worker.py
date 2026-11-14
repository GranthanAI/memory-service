"""app/workers/billing_events_worker.py

Consumes the two topics other services publish about a USER (integration plan 4.3):
  entitlements.changed (billing)  -> drop the user's cached entitlements (both Redis keys), so a plan change is seen at once
  user.deleted         (auth)     -> drop them too, then delete the user's memory data (UserDeletionService)

Deduplicated through processed_events on the event's event_id (at-least-once delivery): a redelivered event does nothing. The offset
is committed only after the handler succeeded; a failed handler leaves the message uncommitted and the consumer rewinds to it, so a
deletion is never dropped (a lost user.deleted is a privacy bug), while a poison message that cannot even be parsed is skipped.
"""

import asyncio
import json
import logging
from typing import Optional

from aiokafka import AIOKafkaConsumer

from app.core.config import settings

logger = logging.getLogger("memory_service.workers.billing_events_worker")


class BillingEventsWorker:
    def __init__(self, entitlements, processed_event_repo, user_deleter=None,
                 bootstrap_servers: str = settings.KAFKA_BOOTSTRAP_SERVERS, group_id: str = "billing-events-worker-group"):
        self.entitlements = entitlements
        self.processed = processed_event_repo
        self.user_deleter = user_deleter          # UserDeletionService (P8c)
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self._consumer = None
        self.is_running = False
        self._task = None

    async def handle(self, topic: str, payload: dict) -> str:
        """One decoded event. Returns what happened (for tests and logs); raises when the effect failed (nothing recorded)."""
        event_id, user_id = str(payload.get("event_id") or ""), str(payload.get("user_id") or "")
        if not event_id or not user_id:
            logger.warning("billing event without event_id or user_id on %s: ignored", topic)
            return "ignored"
        if self.processed.is_event_processed(event_id):
            return "duplicate"
        if topic == settings.ENTITLEMENTS_CHANGED_TOPIC:
            await self.entitlements.invalidate(user_id)
            outcome = "invalidated"
        elif topic == settings.USER_DELETED_TOPIC:
            await self.entitlements.invalidate(user_id)
            if self.user_deleter is not None:
                await self.user_deleter.delete_user(user_id)
            outcome = "user_deleted"
        else:
            return "ignored"
        self.processed.register_event(event_id, user_id)      # only after the effect: a crash in between just repeats it
        return outcome

    async def start(self) -> None:
        topics = (settings.ENTITLEMENTS_CHANGED_TOPIC, settings.USER_DELETED_TOPIC)
        logger.info(f"Starting BillingEventsWorker on {topics}")
        self._consumer = AIOKafkaConsumer(
            *topics, bootstrap_servers=self.bootstrap_servers, group_id=self.group_id, enable_auto_commit=False,
            auto_offset_reset="earliest", session_timeout_ms=settings.KAFKA_SESSION_TIMEOUT_MS,
            max_poll_interval_ms=settings.KAFKA_MAX_POLL_INTERVAL_MS)
        await self._consumer.start()
        self.is_running = True
        self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        while self.is_running:
            try:
                batch = await self._consumer.getmany(timeout_ms=1000)
                for tp, messages in batch.items():
                    for msg in messages:
                        try:
                            payload = json.loads(msg.value.decode("utf-8"))
                            if not isinstance(payload, dict):
                                raise ValueError("not an object")
                        except Exception:  # noqa: BLE001
                            logger.warning("unparseable billing event on %s skipped", tp.topic)
                            continue
                        try:
                            await self.handle(tp.topic, payload)
                        except Exception as e:  # noqa: BLE001
                            logger.error(f"billing event on {tp.topic} failed ({type(e).__name__}: {e}); rewinding to retry")
                            self._consumer.seek(tp, msg.offset)
                            await asyncio.sleep(2.0)
                            break                                   # nothing after a failed message is committed
                    await self._consumer.commit()
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                logger.error(f"BillingEventsWorker polling loop error: {e}")
                await asyncio.sleep(2.0)

    async def stop(self) -> None:
        self.is_running = False
        if self._task:
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None
        if self._consumer:
            await self._consumer.stop()
            self._consumer = None
