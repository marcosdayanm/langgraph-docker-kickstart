"""Application startup: database, Postgres checkpointer/store, and the Deep Agent.

Kept out of `main.py` so the FastAPI routes aren't mixed in with resource wiring.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession

from app.agent.graph import Agent, build_agent, seed_skills
from app.db import create_database, create_tables, seed_catalog
from app.settings import Settings, load_settings


@dataclass(frozen=True, slots=True)
class AppState:
    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    agent: Agent


async def create_app_state(stack: AsyncExitStack) -> AppState:
    """Build the database, durable checkpointer/store, and the agent.

    The checkpointer and store each own a Postgres connection pool that must be
    closed on shutdown. `stack` (owned by the caller's lifespan) tracks both, so
    they close in reverse order without nesting `async with X, Y:` here.
    """
    settings = load_settings()
    engine, session_factory = create_database(settings)
    await create_tables(engine)
    await seed_catalog(session_factory)

    checkpointer = await stack.enter_async_context(
        AsyncPostgresSaver.from_conn_string(settings.database_url)
    )
    store = await stack.enter_async_context(AsyncPostgresStore.from_conn_string(settings.database_url))
    await checkpointer.setup()
    await store.setup()
    await seed_skills(store)

    agent = build_agent(settings, checkpointer, store, session_factory)
    return AppState(settings=settings, engine=engine, session_factory=session_factory, agent=agent)
