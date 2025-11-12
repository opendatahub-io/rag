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

from typing import Any, Dict, List, Optional, Union
from functools import wraps
from fastmcp import FastMCP
from logger import setup_logger
from database_manager import DatabaseManager
import sys
import os
import psycopg
from psycopg.rows import dict_row
from datetime import datetime

logger = setup_logger()
db = DatabaseManager.get_instance()

# Use dict_row factory to get dictionaries instead of tuples
db.cursor = db.conn.cursor(row_factory=dict_row)

mcp = FastMCP("redbank_postgresql")
logger.info("MCP Server initialized: RedBank PostgreSQL")


def validate_date(date_str: str, param_name: str) -> str:
    """Validate date string is in YYYY-MM-DD format

    Args:
        date_str: Date string to validate
        param_name: Parameter name for error messages

    Returns:
        Validated date string

    Raises:
        ValueError: If date format is invalid
    """
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except ValueError:
        raise ValueError(
            f"{param_name} must be in YYYY-MM-DD format (e.g., 2025-01-15), got: {date_str}"
        )


def validate_int(value: Any, param_name: str) -> int:
    """Validate and convert value to integer

    Args:
        value: Value to validate (can be int, str, or other)
        param_name: Parameter name for error messages

    Returns:
        Validated integer value

    Raises:
        ValueError: If value cannot be converted to integer
    """
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"{param_name} must be a valid integer, got: {value}")
    raise ValueError(
        f"{param_name} must be a valid integer type, got: {type(value).__name__}"
    )


def handle_db_errors(func):
    """Handle database errors"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except psycopg.Error as e:
            db.conn.rollback()
            logger.error(f"Database error in {func.__name__}: {e}")
            raise RuntimeError(f"Database error: {str(e)}")
        except ValueError as e:
            logger.error(f"Invalid input in {func.__name__}: {e}")
            raise RuntimeError(f"Invalid input: {str(e)}")

    return wrapper


@mcp.tool()
@handle_db_errors
def get_customer(
    email: Optional[str] = None,
    phone: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Get customer by email or phone number

    Args:
        email: Customer email address
        phone: Customer phone number

    Returns:
        Customer details or empty dict if not found
    """
    if not email and not phone:
        raise ValueError("Either email or phone must be provided")

    field = "email" if email else "phone"
    value = email if email else phone
    logger.info(f"Retrieving customer with {field}: {value}")

    query = f"""
        SELECT customer_id, name, email, phone, address, account_type, 
               date_of_birth, created_date 
        FROM customers 
        WHERE {field} = %s
    """
    db.cursor.execute(query, (value,))
    result = db.cursor.fetchone()

    if not result:
        logger.info(f"No customer found with {field}: {value}")
        return {}

    logger.info(f"Found customer: {result['name']} (ID: {result['customer_id']})")
    return dict(result)


@mcp.tool()
@handle_db_errors
def get_customers_by_name(
    name: str, session_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get customers by name (partial match)

    Args:
        name: Name or partial name to get customers by

    Returns:
        List of matching customers
    """
    logger.info(f"Getting customers with name: {name}")

    query = """
        SELECT customer_id, name, email, phone, address, account_type, 
               date_of_birth, created_date 
        FROM customers 
        WHERE LOWER(name) LIKE LOWER(%s) 
        ORDER BY name
    """
    db.cursor.execute(query, (f"%{name}%",))
    results = [dict(row) for row in db.cursor.fetchall()]

    logger.info(f"Found {len(results)} customers")
    return results


@mcp.tool()
@handle_db_errors
def get_customer_statements(
    customer_id: Union[int, str], session_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get all statements for a customer

    Args:
        customer_id: Customer ID (can be integer or string)

    Returns:
        List of statements
    """
    customer_id = validate_int(customer_id, "customer_id")
    logger.info(f"Retrieving statements for customer: {customer_id}")

    query = """
        SELECT s.statement_id, s.customer_id, c.name as customer_name,
               s.statement_period_start, s.statement_period_end, 
               s.balance, s.created_date
        FROM statements s
        JOIN customers c ON s.customer_id = c.customer_id
        WHERE s.customer_id = %s
        ORDER BY s.statement_period_start DESC
    """
    db.cursor.execute(query, (customer_id,))
    results = [dict(row) for row in db.cursor.fetchall()]

    logger.info(f"Found {len(results)} statements")
    return results


@mcp.tool()
@handle_db_errors
def get_statement_transactions(
    statement_id: Union[int, str], session_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get all transactions for a statement

    Args:
        statement_id: Statement ID (can be integer or string)

    Returns:
        List of transactions
    """
    statement_id = validate_int(statement_id, "statement_id")
    logger.info(f"Retrieving transactions for statement: {statement_id}")

    query = """
        SELECT t.transaction_id, t.statement_id, s.customer_id, 
               c.name as customer_name, t.transaction_date, t.amount, 
               t.description, t.transaction_type, t.merchant
        FROM transactions t
        JOIN statements s ON t.statement_id = s.statement_id
        JOIN customers c ON s.customer_id = c.customer_id
        WHERE t.statement_id = %s
        ORDER BY t.transaction_date DESC
    """
    db.cursor.execute(query, (statement_id,))
    results = [dict(row) for row in db.cursor.fetchall()]

    logger.info(f"Found {len(results)} transactions")
    return results


@mcp.tool()
@handle_db_errors
def get_customer_transactions(
    customer_id: Union[int, str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    session_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get customer transactions with optional date filtering

    Args:
        customer_id: Customer ID (can be integer or string)
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)

    Returns:
        List of transactions
    """
    customer_id = validate_int(customer_id, "customer_id")
    logger.info(f"Retrieving transactions for customer: {customer_id}")

    query = """
        SELECT t.transaction_id, t.statement_id, s.customer_id, 
               c.name as customer_name, t.transaction_date, t.amount, 
               t.description, t.transaction_type, t.merchant
        FROM transactions t
        JOIN statements s ON t.statement_id = s.statement_id
        JOIN customers c ON s.customer_id = c.customer_id
        WHERE s.customer_id = %s
    """
    params = [customer_id]

    if start_date:
        start_date = validate_date(start_date, "start_date")
        query += " AND DATE(t.transaction_date) >= %s"
        params.append(start_date)

    if end_date:
        end_date = validate_date(end_date, "end_date")
        query += " AND DATE(t.transaction_date) <= %s"
        params.append(end_date)

    query += " ORDER BY t.transaction_date DESC"

    db.cursor.execute(query, tuple(params))
    results = [dict(row) for row in db.cursor.fetchall()]

    logger.info(f"Found {len(results)} transactions")
    return results


@mcp.tool()
@handle_db_errors
def get_statement_summary(
    statement_id: Union[int, str], session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get statement summary with balance and transaction stats

    Args:
        statement_id: Statement ID (can be integer or string)

    Returns:
        Statement summary or empty dict if not found
    """
    statement_id = validate_int(statement_id, "statement_id")
    logger.info(f"Getting summary for statement: {statement_id}")

    query = """
        SELECT 
            s.statement_id, s.customer_id, c.name as customer_name,
            s.statement_period_start, s.statement_period_end, s.balance,
            COUNT(t.transaction_id) as total_transactions,
            COUNT(CASE WHEN t.transaction_type = 'CREDIT' THEN 1 END) as credit_count,
            COUNT(CASE WHEN t.transaction_type = 'DEBIT' THEN 1 END) as debit_count,
            COALESCE(SUM(CASE WHEN t.transaction_type = 'CREDIT' THEN ABS(t.amount) END), 0) as credit_total,
            COALESCE(SUM(CASE WHEN t.transaction_type = 'DEBIT' THEN ABS(t.amount) END), 0) as debit_total
        FROM statements s
        JOIN customers c ON s.customer_id = c.customer_id
        LEFT JOIN transactions t ON s.statement_id = t.statement_id
        WHERE s.statement_id = %s
        GROUP BY s.statement_id, s.customer_id, c.name, 
                 s.statement_period_start, s.statement_period_end, s.balance
    """
    db.cursor.execute(query, (statement_id,))
    result = db.cursor.fetchone()

    if not result:
        logger.info(f"No statement found: {statement_id}")
        return {}

    # Rename fields for clarity
    summary = dict(result)
    summary["ending_balance"] = summary.pop("balance")
    summary["credit_transactions"] = summary.pop("credit_count")
    summary["debit_transactions"] = summary.pop("debit_count")
    summary["total_credits"] = summary.pop("credit_total")
    summary["total_debits"] = summary.pop("debit_total")

    logger.info(f"Retrieved summary for statement: {statement_id}")
    return summary


@mcp.tool()
@handle_db_errors
def get_customer_summary(
    customer_id: Union[int, str], session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get customer summary with account info and latest balance

    Args:
        customer_id: Customer ID (can be integer or string)

    Returns:
        Customer summary or empty dict if not found
    """
    customer_id = validate_int(customer_id, "customer_id")
    logger.info(f"Getting summary for customer: {customer_id}")

    query = """
        SELECT 
            c.customer_id, c.name, c.email, c.phone, c.address, 
            c.account_type, c.date_of_birth,
            COUNT(DISTINCT s.statement_id) as total_statements,
            MAX(s.statement_id) as latest_statement_id,
            MAX(s.statement_period_end) as latest_statement_date,
            (SELECT balance FROM statements 
             WHERE customer_id = c.customer_id 
             ORDER BY statement_period_end DESC LIMIT 1) as latest_balance
        FROM customers c
        LEFT JOIN statements s ON c.customer_id = s.customer_id
        WHERE c.customer_id = %s
        GROUP BY c.customer_id, c.name, c.email, c.phone, c.address, 
                 c.account_type, c.date_of_birth
    """
    db.cursor.execute(query, (customer_id,))
    result = db.cursor.fetchone()

    if not result:
        logger.info(f"No customer found: {customer_id}")
        return {}

    logger.info(f"Retrieved summary for customer: {customer_id}")
    return dict(result)


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting RedBank PostgreSQL MCP server on http://{host}:{port}/mcp")

    try:
        mcp.run(transport="http", host=host, port=port, path="/mcp")
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        db.close()
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        db.close()
        sys.exit(1)
