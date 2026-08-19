"""Deep Agent setup, durable memory backend, skills, and response helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.backends.utils import create_file_data
from langchain.agents.structured_output import ToolStrategy
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession

from app.langgraph_helpers import FinalResponse
from app.settings import Settings
from app.tools.customer_memory import read_customer_memory, save_customer_memory
from app.tools.retail import retail_tools
from app.tools.time import current_utc_time

SKILLS_NAMESPACE = ("deep-agent", "skills")
SKILLS_PATH = Path(__file__).parents[1] / "skills"


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Runtime identity used to scope durable user memory."""

    user_id: str


type Agent = CompiledStateGraph[Any, AgentContext, Any, Any]


def build_backend(store: AsyncPostgresStore) -> CompositeBackend:
    """Keep scratch files thread-scoped and make skills/memories durable."""
    return CompositeBackend(
        default=StateBackend(),
        routes={
            "/skills/": StoreBackend(namespace=lambda _runtime: SKILLS_NAMESPACE, store=store),
            "/memories/": StoreBackend(
                namespace=lambda runtime: ("user-memories", runtime.context.user_id), store=store
            ),
        },
    )


async def seed_skills(store: BaseStore) -> None:
    """Make every local SKILL.md available through the durable `/skills/` backend."""
    for skill_file in sorted(SKILLS_PATH.rglob("SKILL.md")):
        key = f"/{skill_file.relative_to(SKILLS_PATH).as_posix()}"
        skill_data = cast(dict[str, Any], create_file_data(skill_file.read_text()))
        await store.aput(SKILLS_NAMESPACE, key, skill_data)


def build_agent(
    settings: Settings,
    checkpointer: AsyncPostgresSaver,
    store: AsyncPostgresStore,
    session_factory: async_sessionmaker[AsyncSession],
) -> Agent:
    model = ChatGoogleGenerativeAI(
        model=settings.vertex_model,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
        vertexai=True,
        temperature=0,
    )
    return create_deep_agent(
        model=model,
        tools=[
            current_utc_time,
            read_customer_memory,
            save_customer_memory,
            *retail_tools(session_factory),
        ],
        checkpointer=checkpointer,
        store=store,
        backend=build_backend(store),
        context_schema=AgentContext,
        skills=["/skills/"],
        permissions=[FilesystemPermission(operations=["write"], paths=["/skills/**"], mode="deny")],
        # Deep Agents accepts ToolStrategy, so the normal REST response remains typed.
        response_format=ToolStrategy(FinalResponse),
        system_prompt=(
            "You are StoreMate, a decisive retail sales assistant. Help customers discover products, "
            "check live inventory, explain pricing, and complete an approved purchase. Use find_articles "
            "for catalog questions. Its SKU and stock fields are internal: never volunteer either. If a "
            "requested quantity cannot be purchased, say there is not enough inventory. Reveal an exact "
            "inventory amount only when the customer explicitly asks for their purchase limit. "
            "Use create_order only after its customer approval interrupt. Keep useful customer facts in "
            "durable memory. When a customer explicitly states a useful, "
            "non-sensitive personal fact or shopping preference, call save_customer_memory before "
            "answering. When asked about that customer or their preferences, call read_customer_memory "
            "before answering; never guess. Do not store payment data, passwords, or full addresses. "
            "Use current_utc_time for time questions. "
            "Finish every normal response with the FinalResponse tool."
        ),
    )


def graph_config(thread_id: UUID) -> RunnableConfig:
    return {"configurable": {"thread_id": str(thread_id)}, "recursion_limit": 20}
