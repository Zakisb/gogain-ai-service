"""Date/Time tools for agents: ISO 8601, Unix, French human-readable, and YYYY-MM-DD.

Backwards compatible: retains `get_current_date` and `get_date_tool`.
Fully migrated to LangChain.
"""

from datetime import datetime, timezone
import time as _time
from langchain.tools import StructuredTool
from typing import Dict


def _french_weekday(dt: datetime) -> str:
    days = [
        "lundi",
        "mardi",
        "mercredi",
        "jeudi",
        "vendredi",
        "samedi",
        "dimanche",
    ]
    # Python weekday: Monday=0
    return days[dt.weekday()]


def _french_month(dt: datetime) -> str:
    months = [
        "janvier",
        "février",
        "mars",
        "avril",
        "mai",
        "juin",
        "juillet",
        "août",
        "septembre",
        "octobre",
        "novembre",
        "décembre",
    ]
    return months[dt.month - 1]


def get_current_date() -> str:
    """Get the current date in YYYY-MM-DD format (backward compatible)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def get_current_datetime_formats() -> Dict[str, str]:
    """Return multiple current datetime formats.

    - iso8601: e.g., 2025-09-30T14:32:15Z
    - unix_timestamp: e.g., 1727704335.123
    - french_human: e.g., mardi 30 septembre 2025, 14h32
    - date_ymd: e.g., 2025-09-30
    """
    now_utc = datetime.now(timezone.utc)
    # ISO 8601 with Z (UTC)
    iso8601 = now_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    # Unix timestamp with milliseconds precision
    unix_timestamp = f"{_time.time():.3f}"
    # French human-readable (UTC time for consistency)
    french_human = f"{_french_weekday(now_utc)} {now_utc.day} {_french_month(now_utc)} {now_utc.year}, {now_utc.strftime('%Hh%M')}"
    # Backward-compatible YYYY-MM-DD
    date_ymd = now_utc.strftime("%Y-%m-%d")
    return {
        "iso8601": iso8601,
        "unix_timestamp": unix_timestamp,
        "french_human": french_human,
        "date_ymd": date_ymd,
    }


def current_datetime_context_line() -> str:
    """Return a single-line context string combining ISO and French formats.

    Format: "Current date and time: 2025-09-30T14:32:15Z (mardi 30 septembre 2025, 14h32)"
    """
    f = get_current_datetime_formats()
    return f"Current date and time: {f['iso8601']} ({f['french_human']})"


def get_date_tool() -> StructuredTool:
    """Factory function to create the legacy date tool (YYYY-MM-DD)."""
    return StructuredTool.from_function(
        func=get_current_date,
        name="get_current_date",
        description="Get today's date in YYYY-MM-DD format. Use this whenever you need the current date."
    )


def get_datetime_formats_tool() -> StructuredTool:
    """Factory creating a tool that returns multiple datetime formats (ISO, Unix, French, YYYY-MM-DD)."""
    return StructuredTool.from_function(
        func=get_current_datetime_formats,
        name="get_current_datetime_formats",
        description=(
            "Obtenir la date/heure actuelle aux formats: ISO 8601 (Z), Unix (secondes.écart), "
            "Français lisible (mardi 30 septembre 2025, 14h32), et YYYY-MM-DD."
        ),
    )


def get_datetime_context_line_tool() -> StructuredTool:
    """Factory creating a tool that returns a combined context line for current datetime."""
    return StructuredTool.from_function(
        func=current_datetime_context_line,
        name="get_current_datetime_context_line",
        description=(
            "Obtenir une ligne de contexte avec la date/heure actuelle: "
            "'Current date and time: <ISO 8601> (<français>)'."
        ),
    )
