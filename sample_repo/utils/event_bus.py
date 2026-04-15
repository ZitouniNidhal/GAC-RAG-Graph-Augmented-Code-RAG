"""Simple in-process event bus for service-to-service communication."""
from typing import Callable, Dict, List


class EventBus:
    """
    Publish/subscribe event bus.
    Services emit events; listeners react asynchronously.
    """
    _listeners: Dict[str, List[Callable]] = {}

    @classmethod
    def subscribe(cls, event: str, handler: Callable) -> None:
        cls._listeners.setdefault(event, []).append(handler)

    @classmethod
    def emit(cls, event: str, payload: dict) -> None:
        """Emit an event. Listeners are called synchronously (demo mode)."""
        for handler in cls._listeners.get(event, []):
            try:
                handler(payload)
            except Exception:
                pass    # Swallow listener errors — events are fire-and-forget

    @classmethod
    def clear(cls) -> None:
        cls._listeners.clear()
