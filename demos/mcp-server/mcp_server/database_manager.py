import psycopg
import threading
from typing import Optional
import logging

logger = logging.getLogger("redbank_financials_db")

class DatabaseManager:
    """
    Singleton DatabaseManager class for interacting with PostgreSQL database
    """
    _instance: Optional['DatabaseManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'DatabaseManager':
        """Ensure only one instance exists (Singleton pattern)"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DatabaseManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize database connection (only once)"""
        if self._initialized:
            return
            
        try:
            self.conn = psycopg.connect(
                host="postgres",
                database="bank_statements",
                user="postgres", 
                password="postgres"
            )
            self.cursor = self.conn.cursor()
            self._initialized = True
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise

    def close(self) -> None:
        """Close database connection"""
        if hasattr(self, 'cursor') and self.cursor:
            self.cursor.close()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        logger.info("Database connection closed")

    @classmethod
    def get_instance(cls) -> 'DatabaseManager':
        """Get the singleton instance"""
        return cls()
