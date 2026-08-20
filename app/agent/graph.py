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

from app.agent.prompt import AGENT_SYSTEM_PROMPT
from app.agent.responses import FinalResponse
from app.agent.tools.customer_memory import read_customer_memory, save_customer_memory
from app.agent.tools.retail import retail_tools
from app.agent.tools.time import current_utc_time
from app.settings import Settings

SKILLS_NAMESPACE = ("deep-agent", "skills")
SKILLS_PATH = Path(__file__).parents[2] / "skills"


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
    """Sync the durable `/skills/` backend with the local SKILL.md files on disk.

    Upserts every skill currently on disk and removes store entries for
    skills that were renamed or deleted, so a stale skill never outlives the
    file it came from.
    """
    disk_skills = {
        f"/{skill_file.relative_to(SKILLS_PATH).as_posix()}": skill_file
        for skill_file in SKILLS_PATH.rglob("SKILL.md")
    }
    existing = await store.asearch(SKILLS_NAMESPACE, limit=1000)
    for item in existing:
        if item.key not in disk_skills:
            await store.adelete(SKILLS_NAMESPACE, item.key)
    for key, skill_file in disk_skills.items():
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
        response_format=ToolStrategy(FinalResponse),
        system_prompt=AGENT_SYSTEM_PROMPT,
    )


def graph_config(thread_id: UUID) -> RunnableConfig:
    return {"configurable": {"thread_id": str(thread_id)}, "recursion_limit": 20}
