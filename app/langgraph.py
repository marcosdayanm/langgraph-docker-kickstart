"""LangGraph agent, tools, and small graph-state helpers."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from langchain.agents import AgentState, create_agent  # type: ignore
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph  # type: ignore
from langgraph.types import Interrupt, StateSnapshot, interrupt
from pydantic import BaseModel, Field, StrictBool

from app.settings import Settings


class FinalResponse(BaseModel):
    """The one validated shape returned to the API after a normal agent turn."""

    answer: str = Field(description="A concise final answer for the user.")


class ApprovalInterrupt(BaseModel):
    """JSON-serializable data surfaced while the graph waits for a human."""

    kind: Literal["approval"] = "approval"
    action: str = Field(min_length=1, max_length=500)


class InterruptBoolValidator(BaseModel):
    """Validated resume value, tied to the dynamic action the model requested."""

    action: str = Field(min_length=1, max_length=500)
    approved: StrictBool


# `create_agent` is generic, but LangGraph's input/output state types are dynamic.
# Keeping them as `Any` at this boundary prevents type-checker noise in FastAPI routes.
type Agent = CompiledStateGraph[AgentState[FinalResponse], Any, Any, Any]


@tool
def current_utc_time() -> str:
    """Return the current UTC time. Use this when the user asks for the time."""
    return datetime.now(UTC).strftime("UTC time: %Y-%m-%d %H:%M:%S")


@tool
def request_approval(action: str) -> str:
    """Pause and ask the user to approve or reject a potentially sensitive action."""
    request = ApprovalInterrupt(action=action)
    decision = InterruptBoolValidator(action=request.action, approved=interrupt(request.model_dump()))
    return decision.model_dump_json()


def build_agent(settings: Settings, checkpointer: AsyncPostgresSaver) -> Agent:
    model = ChatGoogleGenerativeAI(
        model=settings.vertex_model,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
        vertexai=True,
        temperature=0,
    )
    return create_agent(
        model=model,
        tools=[current_utc_time, request_approval],
        checkpointer=checkpointer,
        # ToolStrategy adds a generated `FinalResponse` tool. The model must call it
        # to end every normal turn, so FastAPI receives a validated answer shape.
        # Keep this separate from `@tool(return_direct=True)`: direct-return tools
        # end immediately and intentionally skip this final structured response.
        response_format=ToolStrategy(FinalResponse),
        system_prompt=(
            "You are a concise demo assistant. Use current_utc_time for time questions. "
            "Before simulating an external or sensitive action, call request_approval. "
            "Finish every normal response with the FinalResponse tool."
        ),
    )


def graph_config(thread_id: UUID) -> RunnableConfig:
    return {"configurable": {"thread_id": str(thread_id)}, "recursion_limit": 10}


def interrupt_value(interrupt_item: Interrupt | Mapping[str, object]) -> dict[str, object]:
    value = interrupt_item.value if isinstance(interrupt_item, Interrupt) else interrupt_item.get("value")
    return value if isinstance(value, dict) else {"kind": "input", "value": value} # type: ignore


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content # type: ignore
        )
    return str(content or "")


def message_content(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else message.content # type: ignore
    return content_to_text(content)


def state_interrupt(snapshot: StateSnapshot) -> dict[str, object] | None:
    for task in snapshot.tasks:
        if interrupts := task.interrupts:
            return interrupt_value(interrupts[0])
    return None


def structured_answer(value: object) -> str | None:
    if isinstance(value, FinalResponse):
        return value.answer
    if isinstance(value, Mapping) and isinstance(value.get("answer"), str): # type: ignore
        return value["answer"] # type: ignore
    return None


def final_tool_answer(message: object) -> str | None:
    for tool_call in getattr(message, "tool_calls", []):
        if tool_call.get("name") == FinalResponse.__name__:
            return structured_answer(tool_call.get("args"))
    return None


def persisted_message(message: object) -> tuple[Literal["user", "assistant"], str] | None:
    """Map persisted LangGraph messages to the compact API chat shape."""
    message_type = getattr(message, "type", "")
    content = message_content(message)
    if message_type == "human" and content:
        return "user", content
    if message_type == "ai":
        if answer := final_tool_answer(message):
            return "assistant", answer
        if content:
            return "assistant", content
    return None
