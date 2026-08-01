import time
from types import TracebackType
from typing import Optional, Type

class Timer:
    """
    Context manager to measure the execution latency of code blocks.
    Exposes `elapsed` property in seconds.
    """
    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType]
    ) -> bool:
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        return False  # Do not suppress any exceptions raised within the context
