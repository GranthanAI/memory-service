"""
app/events/topics.py

Kafka topic definitions used by the Memory Service.
"""

from app.core.config import settings

# =============================================================================
# Inbound Topics (Consumed)
# =============================================================================

CHAT_MESSAGE_CREATED = "chat.message.created"
CHAT_RESPONSE_COMPLETED = "chat.response.completed"

CONVERSATION_CREATED = "conversation.created"
CONVERSATION_UPDATED = "conversation.updated"
CONVERSATION_DELETED = "conversation.deleted"

# =============================================================================
# Internal Worker Topics (Produced & Consumed)
# =============================================================================

SUMMARY_REQUEST = settings.KAFKA_SUMMARY_TOPIC
FACT_REQUEST = settings.KAFKA_FACT_TOPIC
EMBEDDING_REQUEST = settings.KAFKA_EMBEDDING_TOPIC
DELETE_REQUEST = settings.KAFKA_DELETE_TOPIC

# =============================================================================
# Dead Letter Queue
# =============================================================================

DLQ = settings.KAFKA_DLQ_TOPIC
