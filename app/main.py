"""FastAPI app: lifecycle, database dependency, and LangGraph routes."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, cast
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.types import Command
from pydantic import BaseModel, Field, StrictBool, field_validator
from sqlalchemy import desc
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db import Conversation, create_database, create_tables, seed_catalog
from app.langgraph import (
    Agent,
    AgentContext,
    build_agent,
    graph_config,
    seed_skills,
)
from app.langgraph_helpers import (
    interrupt_value,
    persisted_message,
    state_interrupt,
    structured_answer,
)
from app.settings import Settings, load_settings


class ThreadCreate(BaseModel):
    title: str = Field(default="New conversation", max_length=100)

    @field_validator("title")
    @classmethod
    def title_cannot_be_blank(cls, value: str) -> str:
        title = value.strip()
        if not title:
            raise ValueError("title cannot be blank")
        return title


class MessageInput(BaseModel):
    message: str = Field(max_length=8_000)

    @field_validator("message")
    @classmethod
    def message_cannot_be_blank(cls, value: str) -> str:
        message = value.strip()
        if not message:
            raise ValueError("message cannot be blank")
        return message


class ResumeInput(BaseModel):
    approved: StrictBool


class ChatResponse(BaseModel):
    thread_id: str
    reply: str | None = None
    interrupt: dict[str, object] | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ThreadState(BaseModel):
    conversation: Conversation
    messages: list[ChatMessage]
    interrupt: dict[str, object] | None = None


SessionFactory = async_sessionmaker[AsyncSession]


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    session_factory = cast(SessionFactory, request.app.state.session_factory)
    async with session_factory() as session:
        yield session


def get_agent(request: Request) -> Agent:
    return cast(Agent, request.app.state.agent)


def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


Session = Annotated[AsyncSession, Depends(get_session)]
Graph = Annotated[Agent, Depends(get_agent)]
AppSettings = Annotated[Settings, Depends(get_settings)]


def graph_result_to_response(thread_id: str, result: Mapping[str, object]) -> ChatResponse:
    interrupts = result.get("__interrupt__")
    if isinstance(interrupts, Sequence) and not isinstance(interrupts, str) and interrupts:
        return ChatResponse(thread_id=thread_id, interrupt=interrupt_value(interrupts[0]))
    return ChatResponse(
        thread_id=thread_id,
        reply=structured_answer(result.get("structured_response")) or "No response generated.",
    )


async def require_conversation(session: AsyncSession, thread_id: UUID) -> Conversation:
    conversation = await session.get(Conversation, thread_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown thread_id")
    return conversation


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    engine, session_factory = create_database(settings)
    await create_tables(engine)
    await seed_catalog(session_factory)
    async with AsyncPostgresSaver.from_conn_string(
        settings.database_url
    ) as checkpointer, AsyncPostgresStore.from_conn_string(settings.database_url) as store:
        await checkpointer.setup()
        await store.setup()
        await seed_skills(store)
        app.state.agent = build_agent(
            settings,
            checkpointer,
            store,
            session_factory,
        )
        app.state.session_factory = session_factory
        app.state.settings = settings
        yield
    await engine.dispose()


app = FastAPI(title="StoreMate Retail Sales Agent", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/threads", response_model=Conversation, status_code=status.HTTP_201_CREATED)
async def create_thread(
    payload: ThreadCreate, session: Session, settings: AppSettings
) -> Conversation:
    conversation = Conversation(title=payload.title, user_id=settings.local_user_id)
    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)
    return conversation


@app.get("/threads", response_model=list[Conversation])
async def list_threads(session: Session) -> list[Conversation]:
    result = await session.exec(select(Conversation).order_by(desc(col(Conversation.created_at))))
    return list(result.all())


@app.get("/threads/{thread_id}", response_model=ThreadState)
async def get_thread(thread_id: UUID, session: Session, agent: Graph) -> ThreadState:
    conversation = await require_conversation(session, thread_id)
    snapshot = await agent.aget_state(graph_config(thread_id))
    messages = [
        ChatMessage(role=parsed[0], content=parsed[1])
        for message in snapshot.values.get("messages", [])
        if (parsed := persisted_message(message)) is not None
    ]
    return ThreadState(conversation=conversation, messages=messages, interrupt=state_interrupt(snapshot))


@app.post("/threads/{thread_id}/messages", response_model=ChatResponse)
async def send_message(
    thread_id: UUID, payload: MessageInput, session: Session, agent: Graph
) -> ChatResponse:
    conversation = await require_conversation(session, thread_id)
    config = graph_config(thread_id)
    if state_interrupt(await agent.aget_state(config)):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This thread is paused; resume it before sending another message.",
        )
    result = await agent.ainvoke(
        {"messages": [("user", payload.message)]},
        config,
        context=AgentContext(user_id=conversation.user_id),
    )
    return graph_result_to_response(str(thread_id), result)


@app.post("/threads/{thread_id}/resume", response_model=ChatResponse)
async def resume_thread(
    thread_id: UUID, payload: ResumeInput, session: Session, agent: Graph
) -> ChatResponse:
    conversation = await require_conversation(session, thread_id)
    config = graph_config(thread_id)
    pending_interrupt = state_interrupt(await agent.aget_state(config))
    if pending_interrupt is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This thread is not paused.")
    conversation.last_interrupt_action = str(pending_interrupt.get("action", "approval"))
    conversation.last_interrupt_approved = payload.approved
    session.add(conversation)
    await session.commit()
    result = await agent.ainvoke(
        Command(resume=payload.approved),
        config,
        context=AgentContext(user_id=conversation.user_id),
    )
    return graph_result_to_response(str(thread_id), result)


web_dist = Path(__file__).parents[1] / "frontend" / "dist"
if web_dist.is_dir():
    app.mount("/", StaticFiles(directory=web_dist, html=True), name="frontend")
