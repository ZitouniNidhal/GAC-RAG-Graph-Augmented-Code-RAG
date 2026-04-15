"""Notification service — sends emails and push notifications."""
from sample_repo.utils.event_bus import EventBus


class NotificationService:
    """Sends user notifications via email and push."""

    def __init__(self):
        self.bus = EventBus()

    def send_receipt(self, user_id: str, transaction_id: str, amount: float) -> None:
        """Email a payment receipt to the user."""
        self.bus.emit("notification.receipt", {
            "user_id": user_id,
            "transaction_id": transaction_id,
            "amount": amount,
        })

    def send_refund_notice(self, user_id: str, amount: float) -> None:
        """Notify user of a refund."""
        self.bus.emit("notification.refund", {
            "user_id": user_id,
            "amount": amount,
        })

    def send_alert(self, user_id: str, message: str) -> None:
        """Send a general alert to the user."""
        self.bus.emit("notification.alert", {
            "user_id": user_id,
            "message": message,
        })
