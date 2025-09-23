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

from typing import Any, Dict, List
from fastmcp import FastMCP
from logger import setup_logger
from database_manager import DatabaseManager
import sys
import psycopg


logger = setup_logger()

db = DatabaseManager.get_instance()
mcp = FastMCP("redbank_financials")
logger.info("MCP Server initialized: RedBank Financials")


@mcp.tool()
def get_user_by_phone(phone_number: str) -> Dict[str, Any]:
    """Get a specific bank user by their phone number
    
    Args:
        phone_number: The phone number of the user to retrieve
    
    Returns:
        Dictionary containing user details (user_id, name, date_of_birth, address, phone_number)
        Returns empty dict if user not found
    """

    logger.info(f"Attempting to retrieve user with phone number: {phone_number}")
    
    try:
        db.cursor.execute("SELECT * FROM users WHERE phone_number = %s", (phone_number,))
        result = db.cursor.fetchone()

        if not result:
            logger.info(f"No user found with phone number: {phone_number}")
            return {}

        user = {
            "user_id": result[0],
            "name": result[1],
            "date_of_birth": result[2].isoformat() if result[2] else None,
            "address": result[3],
            "phone_number": result[4]
        }
        
        logger.info(f"Successfully retrieved user: {user['name']} (ID: {user['user_id']})")
        return user

    except psycopg.Error as e:
        db.conn.rollback()
        logger.error(f"Database error: {e}")
        raise RuntimeError(f"Failed to retrieve user due to database error: {str(e)}")


@mcp.tool()
def get_statements(user_id: int) -> List[Dict[str, Any]]:
    """Get bank statements for a specific user
    
    Args:
        user_id: Required user ID to retrieve statements for
    
    Returns:
        List of dictionaries containing statement details (id, user_id, user_name, date, total)
    """
    
    logger.info(f"Attempting to retrieve statements for user_id: {user_id}")
    
    try:
        query = """
            SELECT s.id, s.user_id, u.name, s.date, s.total 
            FROM statements s
            JOIN users u ON s.user_id = u.user_id
            WHERE s.user_id = %s
            ORDER BY s.date DESC
        """
        db.cursor.execute(query, (user_id,))
        results = db.cursor.fetchall()

        if not results:
            logger.info(f"No statements found for user_id: {user_id}")
            return []

        statements = []
        for result in results:
            statements.append({
                "statement_id": result[0],
                "user_id": result[1],
                "user_name": result[2],
                "date": result[3].isoformat() if result[3] else None,
                "total": float(result[4]) if result[4] else 0.0
            })
        
        logger.info(f"Retrieved {len(statements)} statements for user_id: {user_id}")
        return statements

    except psycopg.Error as e:
        db.conn.rollback()
        logger.error(f"Database error: {e}")
        raise RuntimeError(f"Failed to retrieve statements due to database error: {str(e)}")

    

@mcp.tool()
def get_transactions(statement_id: int) -> List[Dict[str, Any]]:
    """Get transactions for a specific statement
    
    Args:
        statement_id: Required statement ID to retrieve transactions for
    
    Returns:
        List of dictionaries containing transaction details
    
    Raises:
        RuntimeError: If database operation fails
    """
    logger.info(f"Attempting to retrieve transactions for statement_id: {statement_id}")
    
    try:
        query = """
            SELECT t.id, t.statement_id, s.user_id, u.name, t.description, t.price, t.date
            FROM transactions t
            JOIN statements s ON t.statement_id = s.id
            JOIN users u ON s.user_id = u.user_id
            WHERE t.statement_id = %s
            ORDER BY t.date DESC
        """
        db.cursor.execute(query, (statement_id,))
        transactions = db.cursor.fetchall()
        
        if not transactions:
            logger.info(f"No transactions found for statement_id: {statement_id}")
            return []
        
        transaction_list = []
        for transaction in transactions:
            transaction_list.append({
                "transaction_id": transaction[0],
                "statement_id": transaction[1],
                "user_id": transaction[2],
                "user_name": transaction[3],
                "description": transaction[4],
                "price": float(transaction[5]) if transaction[5] else 0.0,
                "date": transaction[6].isoformat() if transaction[6] else None
            })
        
        logger.info(f"Retrieved {len(transaction_list)} transactions for statement_id: {statement_id}")
        return transaction_list
        
    except psycopg.Error as e:
        db.conn.rollback()
        logger.error(f"Database error during transaction retrieval: {e}")
        raise RuntimeError(f"Failed to retrieve transactions due to database error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during transaction retrieval: {e}")
        raise RuntimeError(f"Failed to retrieve transactions: {str(e)}")


if __name__ == "__main__":
    logger.info("Starting RedBank Financials MCP server on http://127.0.0.1:8000/mcp")
    try:
        # this is will be used for Llama Stack
        mcp.run(transport="http", host="127.0.0.1", port=8000, path="/mcp")
        # asyncio.run(mcp.run()) # this is will be used for Cursor testing
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        db.close()
        logger.info("Database connection closed")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        db.close()
        sys.exit(1)
