"""Transaction data model."""
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Transaction:
    """Represents a financial transaction."""
    user_id: str
    amount: float
    currency: str = "USD"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "pending"

    def save(self, conn) -> None:
        """Persist transaction to the database."""
        conn.execute(
            "INSERT INTO transactions (id, user_id, amount, currency, status) VALUES (?,?,?,?,?)",
            (self.id, self.user_id, self.amount, self.currency, self.status),
        )
        self.status = "completed"

    @classmethod
    def load(cls, transaction_id: str, conn) -> Optional["Transaction"]:
        """Load a transaction from the database by ID."""
        row = conn.execute(
            "SELECT id, user_id, amount, currency, status FROM transactions WHERE id=?",
            (transaction_id,),
        ).fetchone()
        if row:
            return cls(id=row[0], user_id=row[1], amount=row[2], currency=row[3], status=row[4])
        return None

    def refund(self, conn) -> None:
        """Mark this transaction as refunded."""
        conn.execute(
            "UPDATE transactions SET status='refunded' WHERE id=?", (self.id,)
        )
        self.status = "refunded"
