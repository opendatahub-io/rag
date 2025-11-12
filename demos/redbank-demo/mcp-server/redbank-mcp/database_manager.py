# Copyright 2025 IBM, Red Hat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import psycopg
import threading
from typing import Optional
import logging
import os

logger = logging.getLogger("redbank_mcp_db")


class DatabaseManager:
    """
    Singleton DatabaseManager class for interacting with PostgreSQL database
    """

    _instance: Optional["DatabaseManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "DatabaseManager":
        """Ensure only one instance exists"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DatabaseManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize database connection"""
        if self._initialized:
            return

        try:
            host = os.getenv("POSTGRES_HOST", "localhost")
            database = os.getenv("POSTGRES_DATABASE", "db")
            user = os.getenv("POSTGRES_USER", "user")
            password = os.getenv("POSTGRES_PASSWORD", "pass")
            port = os.getenv("POSTGRES_PORT", "5432")

            self.conn = psycopg.connect(
                host=host,
                dbname=database,
                user=user,
                password=password,
                port=port,
            )
            self.cursor = self.conn.cursor()
            self._initialized = True
            logger.info(
                f"Database connection established successfully to {host}:{port}/{database}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise

    def close(self) -> None:
        """Close database connection"""
        if hasattr(self, "cursor") and self.cursor:
            self.cursor.close()
        if hasattr(self, "conn") and self.conn:
            self.conn.close()
        logger.info("Database connection closed")

    @classmethod
    def get_instance(cls) -> "DatabaseManager":
        """Get the singleton instance"""
        return cls()
