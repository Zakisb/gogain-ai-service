"""Database client for PostgreSQL using psycopg2 with connection pooling.

This module provides a simple connection pool for accessing the PostgreSQL database
shared with the Node.js service.
"""

import os
import logging
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

# Global connection pool
_connection_pool: Optional[pool.SimpleConnectionPool] = None


def get_database_url() -> str:
    """Get DATABASE_URL from environment and clean it for psycopg2.

    Removes Prisma-specific query parameters like ?schema=public that psycopg2 doesn't support.
    """
    from app.core.config import os
    from dotenv import load_dotenv
    from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

    # Load .env if not already loaded
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
    load_dotenv(dotenv_path=dotenv_path)

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not found in environment variables")

    # Parse the URL
    parsed = urlparse(database_url)

    # Remove Prisma-specific query parameters (schema, etc.)
    if parsed.query:
        query_params = parse_qs(parsed.query)
        # Remove 'schema' parameter (Prisma-specific)
        query_params.pop('schema', None)

        # Rebuild query string without schema parameter
        new_query = urlencode(query_params, doseq=True) if query_params else ''

        # Rebuild URL
        clean_url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment
        ))
        return clean_url

    return database_url


def initialize_pool(minconn: int = 1, maxconn: int = 10) -> None:
    """Initialize the database connection pool.

    Args:
        minconn: Minimum number of connections in the pool
        maxconn: Maximum number of connections in the pool
    """
    global _connection_pool

    if _connection_pool is not None:
        logger.info("Database pool already initialized")
        return

    try:
        database_url = get_database_url()
        _connection_pool = pool.SimpleConnectionPool(
            minconn,
            maxconn,
            database_url
        )
        logger.info(f"✅ Database connection pool initialized (min={minconn}, max={maxconn})")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database pool: {e}")
        raise


def close_pool() -> None:
    """Close all connections in the pool."""
    global _connection_pool

    if _connection_pool is not None:
        _connection_pool.closeall()
        _connection_pool = None
        logger.info("Database connection pool closed")


@contextmanager
def get_db_connection():
    """Context manager to get a database connection from the pool.

    Usage:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users")
            results = cursor.fetchall()

    Yields:
        psycopg2.connection: Database connection
    """
    global _connection_pool

    if _connection_pool is None:
        initialize_pool()

    conn = None
    try:
        conn = _connection_pool.getconn()
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            _connection_pool.putconn(conn)


def execute_query(
    query: str,
    params: Optional[tuple] = None,
    fetch_one: bool = False,
    fetch_all: bool = True
) -> Optional[Any]:
    """Execute a SQL query and return results.

    Args:
        query: SQL query string
        params: Query parameters (optional)
        fetch_one: If True, fetch only one result
        fetch_all: If True, fetch all results (default)

    Returns:
        Query results as dict or list of dicts, or None
    """
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query, params)

        if fetch_one:
            result = cursor.fetchone()
            return dict(result) if result else None
        elif fetch_all:
            results = cursor.fetchall()
            return [dict(row) for row in results]
        else:
            return None


def execute_insert(query: str, params: Optional[tuple] = None) -> Optional[int]:
    """Execute an INSERT query and return the inserted ID.

    Args:
        query: SQL INSERT query string (should include RETURNING id)
        params: Query parameters (optional)

    Returns:
        Inserted record ID or None
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        result = cursor.fetchone()
        return result[0] if result else None


def execute_update(query: str, params: Optional[tuple] = None) -> int:
    """Execute an UPDATE query and return number of affected rows.

    Args:
        query: SQL UPDATE query string
        params: Query parameters (optional)

    Returns:
        Number of affected rows
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        return cursor.rowcount
