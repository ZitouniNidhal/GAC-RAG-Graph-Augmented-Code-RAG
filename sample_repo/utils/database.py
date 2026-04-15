"""Database connection pool utility."""
import os


class DatabasePool:
    """
    Manages a pool of database connections.
    Uses environment variables for configuration.
    """

    _instance = None

    def __init__(self):
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = int(os.getenv("DB_PORT", 5432))
        self.timeout = int(os.getenv("DB_TIMEOUT", 30))
        self._pool = []
        self._connected = False

    def get_connection(self):
        """
        Acquire a connection from the pool.
        Returns None if DB is unreachable — DOES NOT raise.
        This is the root cause of silent failures.
        """
        try:
            if not self._connected:
                self._connect()
            return self._pool[0] if self._pool else None
        except Exception:
            return None          # ← silent failure here

    def _connect(self):
        """Establish connection to the database."""
        # In real code: psycopg2.connect(host=self.host, port=self.port)
        self._connected = True
        self._pool = [MockConnection()]

    def release(self, conn) -> None:
        """Return a connection to the pool."""
        pass

    def close_all(self) -> None:
        """Close all connections in the pool."""
        self._pool.clear()
        self._connected = False


class MockConnection:
    """Stand-in connection for demo purposes."""
    def execute(self, query, params=()):
        return self
    def fetchone(self):
        return None
    def fetchall(self):
        return []
