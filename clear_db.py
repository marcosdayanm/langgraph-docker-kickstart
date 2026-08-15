"""Delete this demo's conversations and LangGraph checkpoints from DATABASE_URL."""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import text

from app.db import create_database
from app.settings import load_settings

TABLES = "conversation, checkpoints, checkpoint_blobs, checkpoint_writes"


async def clear_database() -> None:
    """Truncate only application conversations and LangGraph checkpoint records."""
    engine, _ = create_database(load_settings())
    try:
        async with engine.begin() as connection:
            await connection.execute(text(f"TRUNCATE TABLE {TABLES} RESTART IDENTITY"))
    finally:
        await engine.dispose()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="Confirm this destructive operation.")
    return parser.parse_args()


def main() -> None:
    if not parse_args().yes:
        raise SystemExit("Nothing deleted. Re-run with --yes to clear this demo database.")
    asyncio.run(clear_database())
    print("Deleted conversations and LangGraph checkpoints.")


if __name__ == "__main__":
    main()
