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
    Asserts configuration defaults load and validate.
    """
    assert settings.APP_NAME == "graphgpt-memory-service"
    assert settings.SHORT_TERM_MESSAGE_LIMIT == 20
    assert settings.VECTOR_DIMENSION == 1536
    
    # Verify we can load settings and override them with keyword arguments
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
