class MemoryServiceException(Exception):
    """
    Base exception class for all errors in the Memory Service.
    """
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

class CircuitBreakerOpenException(MemoryServiceException):
    """
    Raised when the circuit breaker for a downstream service is in OPEN state 
    and blocks subsequent service calls.
    """
    def __init__(self, service_name: str, message: str = ""):
        msg = f"Circuit breaker for service '{service_name}' is OPEN."
        if message:
            msg += f" Details: {message}"
        super().__init__(msg)
        self.service_name = service_name

class DeduplicationException(MemoryServiceException):
    """
    Raised when an action is aborted because it has already been processed (e.g. event idempotency check).
    """
    def __init__(self, event_id: str, message: str = ""):
        msg = f"Event '{event_id}' has already been processed."
        if message:
            msg += f" Details: {message}"
        super().__init__(msg)
        self.event_id = event_id

class JobExecutionException(MemoryServiceException):
    """
    Raised when a background worker job fails repeatedly or encounters an un-retryable error state.
    """
    def __init__(self, job_type: str, job_id: str, message: str):
        msg = f"Job '{job_type}' (ID: {job_id}) execution failed: {message}"
        super().__init__(msg)
        self.job_type = job_type
        self.job_id = job_id
