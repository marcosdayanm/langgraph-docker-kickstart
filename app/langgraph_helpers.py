"""Small adapters between LangGraph values and the API response models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from langgraph.types import Interrupt, StateSnapshot
from pydantic import BaseModel, Field


class FinalResponse(BaseModel):
    """The validated response returned by every completed normal turn."""

    answer: str = Field(description="A concise final answer for the user.")


def interrupt_value(interrupt_item: Interrupt | Mapping[str, object]) -> dict[str, object]:
    """Normalize a LangGraph interrupt into JSON the API can return."""
    value = interrupt_item.value if isinstance(interrupt_item, Interrupt) else interrupt_item.get("value")
    return dict(value) if isinstance(value, Mapping) else {"kind": "input", "value": value}


def content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, Mapping) else str(item) for item in content
        )
    return str(content or "")


def state_interrupt(snapshot: StateSnapshot) -> dict[str, object] | None:
    """Return the first pending interrupt, if this thread is paused."""
    for task in snapshot.tasks:
        if interrupts := task.interrupts:
            return interrupt_value(interrupts[0])
    return None


def structured_answer(value: object) -> str | None:
    if isinstance(value, FinalResponse):
        return value.answer
    if isinstance(value, Mapping) and isinstance(value.get("answer"), str):
        return value["answer"]
    return None


def persisted_message(message: object) -> tuple[Literal["user", "assistant"], str] | None:
    """Map persisted Deep Agent messages to the compact API chat shape."""
    message_type = getattr(message, "type", "")
    content = message.get("content") if isinstance(message, Mapping) else getattr(message, "content", "")
    if message_type == "human" and (text := content_to_text(content)):
        return "user", text
    if message_type == "ai":
        for tool_call in getattr(message, "tool_calls", []):
            if (
                isinstance(tool_call, Mapping)
                and tool_call.get("name") == FinalResponse.__name__
                and (answer := structured_answer(tool_call.get("args")))
            ):
                return "assistant", answer
        if text := content_to_text(content):
            return "assistant", text
    return None
