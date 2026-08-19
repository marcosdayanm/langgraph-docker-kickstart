"""A deliberately small, deterministic tool for the retail-sales demo."""

from datetime import UTC, datetime

from langchain.tools import tool


@tool
def current_utc_time() -> str:
    """Return the current UTC time. Use this when the user asks for the time."""
    return datetime.now(UTC).strftime("UTC time: %Y-%m-%d %H:%M:%S")
