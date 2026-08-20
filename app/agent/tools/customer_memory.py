"""Agent-driven, durable customer memory for the retail sales assistant."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Mapping

from langchain.tools import ToolRuntime, tool
from langgraph.store.base import BaseStore

MEMORY_KEY = "profile"
PROFILE_NAMESPACE = "retail-customer"

# BaseStore has no compare-and-swap, so save_customer_fact's read-modify-write
# would otherwise drop a fact when two calls race for the same user. A
# per-user lock is enough because the API runs as a single process.
_user_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


def profile_namespace(user_id: str) -> tuple[str, str]:
    return PROFILE_NAMESPACE, user_id


async def read_customer_facts(store: BaseStore, user_id: str) -> tuple[str, ...]:
    """Read the customer facts previously chosen by the agent for this user."""
    item = await store.aget(profile_namespace(user_id), MEMORY_KEY)
    value = item.value if item and isinstance(item.value, Mapping) else {}
    return tuple(fact for fact in value.get("facts", []) if isinstance(fact, str))


async def save_customer_fact(store: BaseStore, user_id: str, fact: str) -> None:
    """Append one agent-selected fact without duplicate entries."""
    async with _user_locks[user_id]:
        facts = await read_customer_facts(store, user_id)
        normalized = fact.strip()
        updated_facts = (*facts, normalized) if normalized and normalized not in facts else facts
        await store.aput(profile_namespace(user_id), MEMORY_KEY, {"facts": list(updated_facts[-10:])})


@tool
async def read_customer_memory(runtime: ToolRuntime) -> str:
    """Read saved customer facts before answering a personal question."""
    if runtime.store is None:
        return "Customer memory is unavailable."
    user_id = getattr(runtime.context, "user_id", "local-user")
    facts = await read_customer_facts(runtime.store, user_id)
    return "\n".join(facts) if facts else "No saved customer facts."


@tool
async def save_customer_memory(fact: str, runtime: ToolRuntime) -> str:
    """Save a customer fact explicitly provided in this conversation."""
    if runtime.store is None:
        return "Customer memory is unavailable."
    user_id = getattr(runtime.context, "user_id", "local-user")
    await save_customer_fact(runtime.store, user_id, fact)
    return "Saved for future conversations."
