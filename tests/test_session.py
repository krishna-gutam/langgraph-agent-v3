"""Tests for the UI-agnostic backend session layer."""

import pytest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from backend import AgentSession, Overseer, PendingToolCall, estimate_tokens
from langgraph.checkpoint.memory import MemorySaver


@pytest.fixture
def session():
    return AgentSession(thread_id="test", checkpointer=MemorySaver())


def seed(session, *messages):
    """Put messages into the graph state without calling an LLM."""
    session.app.update_state(session.config, {"messages": list(messages)})


# --- state ------------------------------------------------------------------


def test_session_starts_empty(session):
    assert session.messages == []
    assert session.pending_tool_calls() == []
    assert session.is_awaiting_approval is False


def test_config_uses_thread_id(session):
    assert session.config == {"configurable": {"thread_id": "test"}}


def test_switch_and_new_thread(session):
    session.switch_thread("other")
    assert session.thread_id == "other"
    new_id = session.new_thread()
    assert new_id == session.thread_id and new_id != "other"


def test_list_threads_includes_current(session):
    assert session.thread_id in session.list_threads()


# --- history editing --------------------------------------------------------


def test_undo_last_turn(session):
    seed(
        session,
        HumanMessage(content="first"),
        AIMessage(content="reply one"),
        HumanMessage(content="second"),
        AIMessage(content="reply two"),
    )
    assert session.undo_last_turn() is True
    contents = [m.content for m in session.messages]
    assert contents == ["first", "reply one"]


def test_undo_first_turn(session):
    seed(
        session,
        HumanMessage(content="first"),
        AIMessage(content="reply one"),
        HumanMessage(content="second"),
        AIMessage(content="reply two"),
    )
    assert session.undo_first_turn() is True
    contents = [m.content for m in session.messages]
    assert contents == ["second", "reply two"]


def test_undo_with_no_messages(session):
    assert session.undo_last_turn() is False
    assert session.undo_first_turn() is False


def test_clear_history(session):
    seed(session, HumanMessage(content="hi"), AIMessage(content="hello"))
    session.clear_history()
    assert session.messages == []


def test_delete_message(session):
    seed(session, HumanMessage(content="hi"), AIMessage(content="hello"))
    target = session.messages[0]
    session.delete_message(target.id)
    assert [m.content for m in session.messages] == ["hello"]


def test_thread_summary(session):
    seed(session, HumanMessage(content="hi"), AIMessage(content="hello"))
    summary = session.thread_summary(session.thread_id)
    assert summary["last_human"] == "hi"
    assert summary["last_ai"] == "hello"
    assert summary["message_count"] == 2


# --- tool approval ----------------------------------------------------------


def _paused_session(tool_calls):
    """A session whose graph reports a pause before the tools node."""
    session = AgentSession(thread_id="test", checkpointer=MemorySaver())
    snapshot = MagicMock()
    snapshot.next = ("tools",)
    snapshot.values = {"messages": [MagicMock(tool_calls=tool_calls)]}
    session.app = MagicMock()
    session.app.get_state.return_value = snapshot
    return session


def test_pending_tool_calls_exposed():
    session = _paused_session(
        [{"id": "1", "name": "run_bash", "args": {"script": "ls", "justification": "look"}}]
    )
    calls = session.pending_tool_calls()
    assert session.is_awaiting_approval is True
    assert len(calls) == 1
    assert calls[0].name == "run_bash"
    assert calls[0].justification == "look"
    assert calls[0].display_args == {"script": "ls"}


def test_approve_tools_resumes_graph():
    session = _paused_session([{"id": "1", "name": "run_bash", "args": {}}])
    session.approve_tools()
    session.app.invoke.assert_called_once_with(None, config=session.config)


def test_send_tool_feedback_writes_tool_messages():
    session = _paused_session([{"id": "1", "name": "run_bash", "args": {}}])
    session.send_tool_feedback("do it differently")
    _config, update = session.app.update_state.call_args[0]
    assert update["messages"][0].content == "User interrupted: do it differently"
    assert session.app.update_state.call_args[1]["as_node"] == "tools"


def test_deny_tools_removes_last_turn(session):
    seed(
        session,
        HumanMessage(content="first"),
        AIMessage(content="reply one"),
        HumanMessage(content="second"),
    )
    session.deny_tools()
    assert [m.content for m in session.messages] == ["first", "reply one"]


def test_pending_tool_call_without_justification():
    call = PendingToolCall(id="1", name="web_fetch", args={"url": "https://x"})
    assert call.justification is None
    assert call.display_args == {"url": "https://x"}


# --- tokens -----------------------------------------------------------------


def test_estimate_tokens_prefers_usage_metadata():
    msg = AIMessage(content="hi", usage_metadata={"total_tokens": 123, "input_tokens": 100, "output_tokens": 23})
    assert estimate_tokens([msg]) == 123


def test_estimate_tokens_falls_back_to_characters():
    assert estimate_tokens([HumanMessage(content="a" * 100)]) == int(100 // 3.675)


def test_estimate_tokens_empty():
    assert estimate_tokens([]) == 0


# --- overseer ---------------------------------------------------------------


def test_overseer_start_and_stop():
    o = Overseer()
    assert o.active is False
    o.start("  ship it  ")
    assert o.active is True and o.goal == "ship it"
    o.stop()
    assert o.active is False


def test_overseer_advance_records_instruction(monkeypatch):
    monkeypatch.setattr("backend.session.next_instruction", lambda **kw: "open cli.py")
    o = Overseer()
    o.start("ship it")
    assert o.advance([]) == "open cli.py"
    assert o.step == 1
    assert o.instructions == ["open cli.py"]
    assert o.last_instruction == "open cli.py"
    assert o.take_pending() == "open cli.py"
    assert o.take_pending() is None


def test_overseer_reset_clears_log(monkeypatch):
    monkeypatch.setattr("backend.session.next_instruction", lambda **kw: "step")
    o = Overseer()
    o.start("goal")
    o.advance([])
    o.reset()
    assert o.active is False and o.step == 0 and o.instructions == []
    assert o.pending_prompt is None


def test_overseer_propagates_failure(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("no API key")

    monkeypatch.setattr("backend.session.next_instruction", boom)
    o = Overseer()
    o.start("goal")
    with pytest.raises(RuntimeError):
        o.advance([])
