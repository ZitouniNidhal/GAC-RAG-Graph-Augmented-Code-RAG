"""Retry handler with exponential backoff."""
import time
from typing import Callable, Any, Optional


class RetryHandler:
    """
    Retries a callable up to max_attempts times with exponential backoff.
    Returns None (silently) if all attempts fail.
    """

    def __init__(self, max_attempts: int = 3, backoff_factor: float = 2.0):
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor

    def attempt(self, fn: Callable, *args, **kwargs) -> Optional[Any]:
        """
        Try calling fn(*args, **kwargs) up to max_attempts times.
        Returns the result on success, None on total failure.
        """
        wait = 1.0
        for attempt_num in range(self.max_attempts):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if attempt_num < self.max_attempts - 1:
                    time.sleep(wait)
                    wait *= self.backoff_factor
        return None          # ← silent failure propagated upward
