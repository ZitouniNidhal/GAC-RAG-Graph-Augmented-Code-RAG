"""Account management service."""
from sample_repo.utils.database import DatabasePool
from sample_repo.utils.event_bus import EventBus


class AccountService:
    """Manages user account balances and status."""

    def __init__(self):
        self.db = DatabasePool()
        self.bus = EventBus()

    def debit(self, user_id: str, amount: float) -> bool:
        """Deduct amount from user balance. Returns True on success."""
        conn = self.db.get_connection()
        if not conn:
            return False
        # Update balance in DB
        conn.execute(
            "UPDATE accounts SET balance = balance - ? WHERE user_id = ?",
            (amount, user_id),
        )
        self.bus.emit("account.debited", {"user_id": user_id, "amount": amount})
        return True

    def flag_failed_payment(self, user_id: str) -> None:
        """Mark a user account as having a failed payment."""
        self.bus.emit("account.payment_failed", {"user_id": user_id})

    def get_balance(self, user_id: str) -> float:
        """Return current balance for a user."""
        conn = self.db.get_connection()
        if not conn:
            return 0.0
        row = conn.execute(
            "SELECT balance FROM accounts WHERE user_id = ?", (user_id,)
        ).fetchone()
        return row[0] if row else 0.0

    def update_status(self, user_id: str, status: str) -> None:
        """Update account status (active, suspended, closed)."""
        conn = self.db.get_connection()
        if not conn:
            return
        conn.execute(
            "UPDATE accounts SET status = ? WHERE user_id = ?",
            (status, user_id),
        )
        self.bus.emit("account.status_changed", {"user_id": user_id, "status": status})
