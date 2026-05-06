"""
Database connection utilities using SQLAlchemy.
"""

from typing import Optional, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from config import get_settings


def create_db_engine() -> Engine:
    """
    Create and return a SQLAlchemy engine.
    Raises errors with useful messages for UI display.
    """
    settings = get_settings()
    db_url = settings.sqlalchemy_url()

    try:
        # pool_pre_ping=True helps keep stale connections healthy.
        engine = create_engine(db_url, pool_pre_ping=True)
        return engine
    except SQLAlchemyError as exc:
        raise RuntimeError(f"Failed to create database engine: {exc}") from exc


def test_connection(engine: Engine) -> Tuple[bool, str]:
    """
    Test DB connectivity with a lightweight query.
    Returns (status, message).
    """
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True, "Database connection successful."
    except SQLAlchemyError as exc:
        return False, f"Database connection failed: {exc}"


def safe_dispose(engine: Optional[Engine]) -> None:
    """
    Close connection pool safely.
    Useful when app exits or reconnects.
    """
    if engine is None:
        return
    try:
        engine.dispose()
    except SQLAlchemyError:
        # Ignore dispose errors because app can continue.
        pass
