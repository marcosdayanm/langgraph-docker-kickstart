import asyncio

import pytest
from langchain_core.messages import AIMessage
from langgraph.store.memory import InMemoryStore

from app.agent.graph import SKILLS_NAMESPACE, seed_skills
from app.agent.responses import persisted_message
from app.agent.tools.customer_memory import read_customer_facts, save_customer_fact
from app.main import graph_result_to_response


@pytest.mark.parametrize(
    ("result", "reply", "interrupt"),
    [
        (
            {"structured_response": {"answer": "The UTC time is 12:00."}},
            "The UTC time is 12:00.",
            None,
        ),
        ({"__interrupt__": [{"value": {"kind": "approval", "action": "deploy"}}]}, None, {"kind": "approval", "action": "deploy"}),
    ],
)
def test_graph_result_to_response_handles_terminal_states(result, reply, interrupt) -> None:
    response = graph_result_to_response("thread-1", result)

    assert response.thread_id == "thread-1"
    assert response.reply == reply
    assert response.interrupt == interrupt


def test_persisted_message_reads_the_final_response_tool_call() -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "FinalResponse", "args": {"answer": "Structured and durable."}, "id": "1"}
        ],
    )

    persisted = persisted_message(message)
    assert persisted is not None
    assert persisted == ("assistant", "Structured and durable.")


def test_seed_skills_makes_the_local_skill_available() -> None:
    store = InMemoryStore()

    asyncio.run(seed_skills(store))

    skill = asyncio.run(store.aget(SKILLS_NAMESPACE, "/retail-sales/SKILL.md"))
    assert skill is not None
    assert "current_utc_time" in skill.value["content"]


def test_agent_saved_customer_fact_is_available_to_a_new_conversation() -> None:
    store = InMemoryStore()

    async def profile_after_first_chat() -> str:
        await save_customer_fact(store, "local-user", "Customer's name is Justin.")
        return "\n".join(await read_customer_facts(store, "local-user"))

    assert asyncio.run(profile_after_first_chat()) == "Customer's name is Justin."
