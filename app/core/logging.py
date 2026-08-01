import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any, Dict, Optional

# Async-safe context variables for trace metadata
var_trace_id: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
var_conversation_id: ContextVar[Optional[str]] = ContextVar("conversation_id", default=None)
var_event_id: ContextVar[Optional[str]] = ContextVar("event_id", default=None)
var_summary_version: ContextVar[Optional[int]] = ContextVar("summary_version", default=None)

def set_log_context(
    trace_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    event_id: Optional[str] = None,
    summary_version: Optional[int] = None
) -> None:
    """
    Sets trace context variables for the current execution context.
    """
    if trace_id is not None:
        var_trace_id.set(trace_id)
    if conversation_id is not None:
        var_conversation_id.set(conversation_id)
    if event_id is not None:
        var_event_id.set(event_id)
    if summary_version is not None:
        var_summary_version.set(summary_version)

def clear_log_context() -> None:
    """
    Clears all context variables for the current execution context.
    """
    var_trace_id.set(None)
    var_conversation_id.set(None)
    var_event_id.set(None)
    var_summary_version.set(None)

class ContextFilter(logging.Filter):
    """
    Logging filter that injects async-safe trace context variables into the LogRecord.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = var_trace_id.get()
        record.conversation_id = var_conversation_id.get()
        record.event_id = var_event_id.get()
        record.summary_version = var_summary_version.get()
        return True

class JSONFormatter(logging.Formatter):
    """
    Formatter that outputs logs in JSON format with injected trace contexts.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "file": f"{record.pathname}:{record.lineno}",
        }
        
        # Inject metadata if present
        if getattr(record, "trace_id", None):
            log_data["trace_id"] = record.trace_id
        if getattr(record, "conversation_id", None):
            log_data["conversation_id"] = record.conversation_id
        if getattr(record, "event_id", None):
            log_data["event_id"] = record.event_id
        if getattr(record, "summary_version", None):
            log_data["summary_version"] = record.summary_version

        # Inject exception details if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)

class ConsoleFormatter(logging.Formatter):
    """
    Human-readable console formatter for local development, with colored trace prefixes.
    """
    def format(self, record: logging.LogRecord) -> str:
        # Resolve trace elements
        trace_id = getattr(record, "trace_id", None) or "-"
        conversation_id = getattr(record, "conversation_id", None) or "-"
        event_id = getattr(record, "event_id", None) or "-"
        summary_ver = getattr(record, "summary_version", None) or "-"

        trace_prefix = f"[{trace_id}][{conversation_id}][{event_id}][v{summary_ver}]"
        
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        log_message = f"{timestamp} [{record.levelname}] {trace_prefix} {record.name}: {record.getMessage()}"
        
        if record.exc_info:
            log_message += f"\n{self.formatException(record.exc_info)}"
            
        return log_message

def setup_logging(debug_mode: bool = False) -> None:
    """
    Configures root logging. Switches formatter between JSON (production) 
    and Console (development) based on debug mode parameters.
    """
    root_logger = logging.getLogger()
    
    # Clean previous handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set log level
    log_level = logging.DEBUG if debug_mode else logging.INFO
    root_logger.setLevel(log_level)

    # Create handler routing to standard stdout stream
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    # Attach ContextFilter
    handler.addFilter(ContextFilter())

    # Attach appropriate Formatter
    if debug_mode:
        handler.setFormatter(ConsoleFormatter())
    else:
        handler.setFormatter(JSONFormatter())

    root_logger.addHandler(handler)
