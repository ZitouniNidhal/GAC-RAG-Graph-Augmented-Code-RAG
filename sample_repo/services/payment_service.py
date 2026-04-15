"""
Payment processing service.
Handles charging customers and recording transactions.
"""
from sample_repo.utils.database import DatabasePool
from sample_repo.utils.retry import RetryHandler
from sample_repo.models.transaction import Transaction
from sample_repo.services.notification_service import NotificationService
from sample_repo.services.account_service import AccountService


class PaymentService:
    """Processes payments and manages the payment lifecycle."""

    def __init__(self):
        self.db = DatabasePool()
        self.retry = RetryHandler(max_attempts=3)
        self.notifier = NotificationService()
        self.account_service = AccountService()

    def process(self, user_id: str, amount: float, currency: str = "USD") -> dict:
        """
        Process a payment for a user.
        Returns a result dict with status and transaction_id.
        """
        conn = self.db.get_connection()
        if not conn:
            # BUG: silently returns None instead of raising
            return None

        def _charge():
            txn = Transaction(user_id=user_id, amount=amount, currency=currency)
            txn.save(conn)
            return txn

        txn = self.retry.attempt(_charge)
        if txn:
            self.account_service.debit(user_id, amount)
            self.notifier.send_receipt(user_id, txn.id, amount)
            return {"status": "success", "transaction_id": txn.id}

        self.account_service.flag_failed_payment(user_id)
        return {"status": "failed", "transaction_id": None}

    def refund(self, transaction_id: str) -> bool:
        """Reverse a completed transaction."""
        conn = self.db.get_connection()
        if not conn:
            return False
        txn = Transaction.load(transaction_id, conn)
        if txn and txn.status == "completed":
            txn.refund(conn)
            self.notifier.send_refund_notice(txn.user_id, txn.amount)
            return True
        return False
