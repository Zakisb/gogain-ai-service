"""Database helpers built on top of Prisma.

This module preserves the legacy helper signatures (execute_query, execute_insert,
etc.) while delegating all database access through the shared Prisma client.  A
dedicated background event loop keeps the Prisma connection warm, allowing the
rest of the codebase to continue using synchronous helpers from worker threads
without blocking the main asyncio loop.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

from prisma import Prisma

from app.core.prisma_client import (
    disconnect_client,
    ensure_client_connected,
    run_async,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #

def _convert_placeholders(query: str, params: Sequence[Any]) -> Tuple[str, Sequence[Any]]:
    """Convert legacy %s placeholders to PostgreSQL $1..$n format."""
    if not params:
        return query, params

    parts = query.split("%s")
    if len(parts) == 1:
        return query, params

    rebuilt: List[str] = []
    for idx, chunk in enumerate(parts):
        rebuilt.append(chunk)
        if idx < len(parts) - 1:
            rebuilt.append(f"${idx + 1}")

    return "".join(rebuilt), params


async def _query_raw(
    client: Prisma,
    query: str,
    params: Sequence[Any],
) -> List[Dict[str, Any]]:
    stmt, values = _convert_placeholders(query, params)
    if values:
        result = await client.query_raw(stmt, *values)
    else:
        result = await client.query_raw(stmt)
    return list(result or [])


async def _execute_raw(
    client: Prisma,
    query: str,
    params: Sequence[Any],
) -> int:
    stmt, values = _convert_placeholders(query, params)
    if values:
        return await client.execute_raw(stmt, *values)
    return await client.execute_raw(stmt)


async def _execute_insert(
    client: Prisma,
    query: str,
    params: Sequence[Any],
) -> Optional[Any]:
    rows = await _query_raw(client, query, params)
    if not rows:
        return None
    first_row = rows[0]
    if not first_row:
        return None
    # Return the first column value (matches legacy psycopg2 behaviour)
    return next(iter(first_row.values()))


async def _execute_query(
    client: Prisma,
    query: str,
    params: Sequence[Any],
    fetch_one: bool,
    fetch_all: bool,
) -> Optional[Any]:
    rows = await _query_raw(client, query, params)
    if fetch_one:
        return rows[0] if rows else None
    if fetch_all:
        return rows
    return None


def _run_coroutine(coro: Awaitable[T]) -> T:
    ensure_client_connected()
    return run_async(coro)


# --------------------------------------------------------------------------- #
# Public helpers (compatibility layer)                                        #
# --------------------------------------------------------------------------- #

def initialize_pool(minconn: int = 5, maxconn: int = 50) -> None:
    """Maintain backwards compatibility with legacy API."""
    ensure_client_connected()
    logger.info("✅ Prisma client ready (pool parameters ignored)")


def close_pool() -> None:
    """Close Prisma client and background loop."""
    disconnect_client()


@contextmanager
def get_db_connection():
    """Connection context placeholder for legacy scripts.

    Prisma hides individual connections; attempting to request a raw connection
    is not supported.  For backwards-compatibility we raise a clear error when
    legacy code still attempts to use low-level cursors.
    """
    raise RuntimeError(
        "Direct connection access is no longer supported. Please refactor the "
        "caller to use app.core.database helpers (execute_query, run_transaction, "
        "etc.) instead of raw cursors."
    )
    yield  # pragma: no cover


def execute_query(
    query: str,
    params: Optional[Sequence[Any]] = None,
    fetch_one: bool = False,
    fetch_all: bool = True,
) -> Optional[Any]:
    params = tuple(params or ())
    client = ensure_client_connected()
    return _run_coroutine(_execute_query(client, query, params, fetch_one, fetch_all))


def execute_insert(
    query: str,
    params: Optional[Sequence[Any]] = None,
) -> Optional[Any]:
    params = tuple(params or ())
    client = ensure_client_connected()
    return _run_coroutine(_execute_insert(client, query, params))


def execute_update(
    query: str,
    params: Optional[Sequence[Any]] = None,
) -> int:
    params = tuple(params or ())
    client = ensure_client_connected()
    return _run_coroutine(_execute_raw(client, query, params))


class TransactionAdapter:
    """Helper passed into run_transaction callbacks."""

    def __init__(self, tx_client):
        self._tx = tx_client

    async def query(
        self,
        query: str,
        params: Optional[Sequence[Any]] = None,
        fetch_one: bool = False,
        fetch_all: bool = True,
    ) -> Optional[Any]:
        return await _execute_query(
            self._tx, query, tuple(params or ()), fetch_one, fetch_all
        )

    async def insert(
        self,
        query: str,
        params: Optional[Sequence[Any]] = None,
    ) -> Optional[Any]:
        return await _execute_insert(self._tx, query, tuple(params or ()))

    async def execute(
        self,
        query: str,
        params: Optional[Sequence[Any]] = None,
    ) -> int:
        return await _execute_raw(self._tx, query, tuple(params or ()))


def run_transaction(
    callback: Callable[[TransactionAdapter], Awaitable[T]],
) -> T:
    """Execute the callback inside a Prisma transaction."""

    async def _runner() -> T:
        client = ensure_client_connected()
        async with client.tx() as tx:
            adapter = TransactionAdapter(tx)
            return await callback(adapter)

    return _run_coroutine(_runner())


# --------------------------------------------------------------------------- #
# Query helpers used by the application                                       #
# --------------------------------------------------------------------------- #

def get_database_url() -> str:
    """Keep legacy helper for diagnostics."""
    from dotenv import load_dotenv

    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not found in environment variables")
    return database_url


def get_user_profile(user_id: int) -> Optional[Dict[str, Any]]:
    """Fetch user profile from database."""
    try:
        query = """
            SELECT
                u.id,
                u.email,
                u."firstName",
                u."lastName",
                u.avatar,
                u.onboarded,
                u."createdAt",
                up."birthDate",
                up.gender,
                up.height,
                up.weight,
                up.goal,
                up."fitnessLevel",
                up."availableEquipments",
                up."desiredTrainingFrequency",
                up."heightUnit",
                up."weightUnit"
            FROM "User" u
            LEFT JOIN "UserProfile" up ON u.id = up."userId"
            WHERE u.id = %s AND u.deleted = false
            LIMIT 1
        """
        return execute_query(query, (user_id,), fetch_one=True)
    except Exception as exc:
        logger.error(f"Failed to fetch user profile for user_id={user_id}: {exc}")
        return None


def get_user_training_program_overview(user_id: int) -> Optional[Dict[str, Any]]:
    """Fetch the most recent active training program with days and exercises."""
    try:
        def _to_iso(value: Any) -> Optional[str]:
            if value is None:
                return None
            return value.isoformat() if hasattr(value, "isoformat") else str(value)

        query = """
            SELECT
                utp.id,
                utp.name,
                utp.description,
                utp.duration,
                utp.frequency,
                utp."startDate",
                utp."endDate",
                utp."createdAt",
                utp.status AS program_status,
                tpf.name AS focus_name,
                td.id AS day_id,
                td."dayNumber" AS day_number,
                td.name AS day_name,
                td.date AS day_date,
                td.status AS day_status,
                upe.id AS exercise_id,
                e.title AS exercise_name,
                e.description AS exercise_description,
                upe."setsPlanned" AS sets_planned,
                upe."repsPerSet" AS reps_per_set,
                upe."restTimeBetweenSets" AS rest_time_seconds
            FROM "UserTrainingProgram" utp
            LEFT JOIN "TrainingProgramFocus" tpf ON utp."focusId" = tpf.id
            LEFT JOIN "UserTrainingDay" td ON utp.id = td."userProgramId"
            LEFT JOIN "UserProgramExercise" upe ON td.id = upe."userTrainingDayId"
            LEFT JOIN "Exercise" e ON upe."exerciseId" = e.id
            WHERE utp."userId" = %s AND utp.status = 'ACTIVE'
            ORDER BY utp."createdAt" DESC, td."dayNumber", upe."orderInDay", upe.id
            LIMIT 1000
        """

        rows = execute_query(query, (user_id,))
        if not rows:
            return None

        program_row = rows[0]
        program = {
            "programId": program_row["id"],
            "name": program_row.get("name"),
            "description": program_row.get("description"),
            "focus": program_row.get("focus_name"),
            "status": program_row.get("program_status"),
            "durationWeeks": program_row.get("duration"),
            "frequencyPerWeek": program_row.get("frequency"),
            "startDate": _to_iso(program_row.get("startDate")),
            "endDate": _to_iso(program_row.get("endDate")),
            "createdAt": _to_iso(program_row.get("createdAt")),
            "trainingDays": [],
        }

        day_map: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            day_id = row.get("day_id")
            if day_id is None:
                continue

            day_entry = day_map.get(day_id)
            if not day_entry:
                day_entry = {
                    "dayId": day_id,
                    "dayNumber": row.get("day_number"),
                    "name": row.get("day_name"),
                    "date": _to_iso(row.get("day_date")),
                    "status": row.get("day_status"),
                    "exercises": [],
                }
                day_map[day_id] = day_entry
                program["trainingDays"].append(day_entry)

            exercise_id = row.get("exercise_id")
            if exercise_id is not None:
                exercise_entry = {
                    "exerciseId": exercise_id,
                    "name": row.get("exercise_name"),
                    "description": row.get("exercise_description"),
                    "sets": row.get("sets_planned"),
                    "reps": row.get("reps_per_set"),
                    "restSeconds": row.get("rest_time_seconds"),
                }
                day_entry["exercises"].append(exercise_entry)

        return program
    except Exception as exc:
        logger.error(f"Failed to fetch training program for user_id={user_id}: {exc}")
        return None
