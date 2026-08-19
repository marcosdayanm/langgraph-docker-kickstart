"""Delete this demo's conversations, checkpoints, and local customer memory."""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import text

from app.db import create_database
from app.settings import load_settings

TABLES = "conversations, checkpoints, checkpoint_blobs, checkpoint_writes, orders, order_items"


async def clear_database() -> None:
    """Delete only the demo's application, checkpoint, and customer-memory records."""
    engine, _ = create_database(load_settings())
    try:
        async with engine.begin() as connection:
            await connection.execute(text(f"TRUNCATE TABLE {TABLES} RESTART IDENTITY"))
            await connection.execute(
                text(
                    "DELETE FROM store WHERE prefix LIKE 'retail-customer.%' "
                    "OR prefix LIKE 'user-memories.%'"
                )
            )
            await connection.execute(
                text("UPDATE products SET stock_quantity = initial_stock_quantity")
            )
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
    print("Deleted conversations, orders, checkpoints, and local customer memories.")


if __name__ == "__main__":
    main()
