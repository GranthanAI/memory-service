"""
app/models/snapshot.py

Lightweight ConversationSnapshot domain representation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from app.models.memory import MemoryState


@dataclass
class ConversationSnapshot:
    """
    Lightweight state metadata for a conversation.
    Does NOT contain recent_messages payload.
    Recent message sliding window is managed separately in Redis and Cassandra.
    """
    conversation_id: str
    user_id: str
    message_count: int
    state: MemoryState
    summary_version: int
    fact_version: int
    snapshot_version: int
    last_summary_msg_id: Optional[str]
    updated_at: datetime
