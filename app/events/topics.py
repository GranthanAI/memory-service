"""
app/events/topics.py

Constants and names for Kafka event topics used by the Memory Service.
"""

from app.core.config import settings

# Inbound message ingest topics
CHAT_MESSAGE_CREATED = "chat.message.created"
CHAT_RESPONSE_COMPLETED = "chat.response.completed"

# Outbound/Pipeline topics
SUMMARY_REQUEST = settings.KAFKA_SUMMARY_TOPIC
FACT_REQUEST = settings.KAFKA_FACT_TOPIC
EMBEDDING_REQUEST = settings.KAFKA_EMBEDDING_TOPIC
DELETE_REQUEST = settings.KAFKA_DELETE_TOPIC
DLQ = settings.KAFKA_DLQ_TOPIC
