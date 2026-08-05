import json
import logging
import pytest
from app.core.config import settings, SystemSettings
from app.core.exceptions import (
    CircuitBreakerOpenException,
    DeduplicationException,
    JobExecutionException,
    MemoryServiceException
)
from app.core.logging import (
    setup_logging,
    set_log_context,
    clear_log_context,
    var_trace_id,
    ConsoleFormatter,
    JSONFormatter
)

def test_system_settings_loading():
    """
    Asserts all configuration groups load with correct defaults.
    Covers all settings groups added in Phase 2 (LLD v3.1).
    """
    # Core
    assert settings.APP_NAME == "graphgpt-memory-service"
    assert settings.HTTP_PORT == 8000

    # Cassandra (new — primary source of truth)
    assert isinstance(settings.CASSANDRA_HOSTS, str)
    assert settings.CASSANDRA_PORT == 9042
    assert settings.CASSANDRA_KEYSPACE == "graphgpt_memory"
    assert settings.CASSANDRA_TIMEOUT_SECONDS == 5.0

    # Redis hot cache
    assert settings.SHORT_TERM_MESSAGE_LIMIT == 20
    assert settings.IDEMPOTENCY_TTL_SECONDS == 604800    # 7 days
    assert settings.REDIS_LOCK_TTL_SECONDS == 5
    assert settings.REDIS_LOCK_WATCHDOG_INTERVAL == 2.0

    # Milvus
    assert settings.VECTOR_DIMENSION == 384
    assert settings.MILVUS_BULK_INSERT_BATCH_SIZE == 100

    # gRPC pool (new)
    assert settings.GRPC_POOL_SIZE == 5
    assert settings.GRPC_HEALTH_CHECK_INTERVAL_SECONDS == 30.0

    # Graph Service (new — graceful fallback)
    assert settings.GRAPH_SERVICE_TIMEOUT_MS == 200

    # Outbox (new — configurable polling)
    assert settings.OUTBOX_POLL_INTERVAL_MS == 1000
    assert settings.OUTBOX_BATCH_SIZE == 50
    assert settings.OUTBOX_STALE_PROCESSING_MINUTES == 5

    # Circuit Breaker (new)
    assert settings.CB_FAILURE_THRESHOLD == 5
    assert settings.CB_RECOVERY_TIMEOUT_SECONDS == 60.0
    assert settings.CB_HALF_OPEN_LIMIT == 2

    # Retrieval scoring
    assert settings.RETRIEVAL_TOP_K_FACTS == 10
    assert settings.FACT_MERGE_SIMILARITY_THRESHOLD == 0.85

    # Custom override still works
    custom = SystemSettings(APP_NAME="custom-name", SHORT_TERM_MESSAGE_LIMIT=40)
    assert custom.APP_NAME == "custom-name"
    assert custom.SHORT_TERM_MESSAGE_LIMIT == 40


def test_custom_exceptions():
    """
    Asserts exceptions serialize metadata variables properly.
    """
    with pytest.raises(CircuitBreakerOpenException) as exc:
        raise CircuitBreakerOpenException("graph-service", "Timeout error")
    assert exc.value.service_name == "graph-service"
    assert "graph-service" in str(exc.value)

    with pytest.raises(DeduplicationException) as exc:
        raise DeduplicationException("evt-123", "Already in database")
    assert exc.value.event_id == "evt-123"
    assert "evt-123" in str(exc.value)

    with pytest.raises(JobExecutionException) as exc:
        raise JobExecutionException("summarize", "job-abc", "Database connection lost")
    assert exc.value.job_type == "summarize"
    assert exc.value.job_id == "job-abc"
    assert "Database connection lost" in str(exc.value)

def test_context_vars_and_logging(capsys):
    """
    Verifies that structured context variables bind and print logs correctly.
    """
    # 1. Setup development / console logging format
    setup_logging(debug_mode=True)
    logger = logging.getLogger("test_console_logger")
    
    set_log_context(
        trace_id="t-1",
        conversation_id="c-2",
        event_id="e-3",
        summary_version=5
    )
    
    assert var_trace_id.get() == "t-1"
    
    logger.info("Test console context message")
    
    captured = capsys.readouterr()
    assert "t-1" in captured.out
    assert "c-2" in captured.out
    assert "e-3" in captured.out
    assert "v5" in captured.out
    assert "Test console context message" in captured.out

    # 2. Setup production / JSON logging format
    setup_logging(debug_mode=False)
    logger_json = logging.getLogger("test_json_logger")
    
    logger_json.info("Test json context message")
    
    captured_json = capsys.readouterr()
    
    # Parse output back to JSON to check keys
    log_json = json.loads(captured_json.out.strip())
    assert log_json["level"] == "INFO"
    assert log_json["trace_id"] == "t-1"
    assert log_json["conversation_id"] == "c-2"
    assert log_json["event_id"] == "e-3"
    assert log_json["summary_version"] == 5
    assert log_json["message"] == "Test json context message"

    # 3. Test clear context
    clear_log_context()
    logger_json.info("After clear message")
    captured_clear = capsys.readouterr()
    log_clear_json = json.loads(captured_clear.out.strip())
    
    assert "trace_id" not in log_clear_json
    assert "conversation_id" not in log_clear_json
    assert log_clear_json["message"] == "After clear message"
