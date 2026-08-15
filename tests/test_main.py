import pytest
from langchain_core.messages import AIMessage

from app.langgraph import InterruptBoolValidator, persisted_message
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


def test_interrupt_bool_validator_keeps_the_custom_action() -> None:
    decision = InterruptBoolValidator(action="deploy the demo", approved=True)

    assert decision.model_dump() == {"action": "deploy the demo", "approved": True}
