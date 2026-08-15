"""The one application-owned PostgreSQL table and its async session factory."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlmodel import Field, SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.settings import Settings


class Conversation(SQLModel, table=True):
    thread_id: UUID = Field(default_factory=uuid4, primary_key=True)
    title: str = Field(max_length=100)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_interrupt_action: str | None = Field(default=None, max_length=500)
    last_interrupt_approved: bool | None = Field(default=None)


def create_database(settings: Settings) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(settings.sqlalchemy_url)
    sessions = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return engine, sessions


async def create_tables(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)
        # `create_all()` does not add columns to an existing POC database. Keep this
        # tiny Postgres-only migration instead of introducing Alembic for two fields.
        await connection.execute(
            text("ALTER TABLE conversation ADD COLUMN IF NOT EXISTS last_interrupt_action VARCHAR(500)")
        )
        await connection.execute(
            text("ALTER TABLE conversation ADD COLUMN IF NOT EXISTS last_interrupt_approved BOOLEAN")
        )
