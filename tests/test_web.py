"""Tests for the FastAPI frontend.

The session is swapped for one backed by MemorySaver, so nothing here touches
the real workspace database.
"""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from backend import AgentSession, PendingToolCall
from frontends.web import api


@pytest.fixture
def client(monkeypatch):
    session = AgentSession(thread_id="test", checkpointer=MemorySaver())
    monkeypatch.setattr(api, "_session", session)
    api._overseer.reset()
    with TestClient(api.app) as client:
        client.session = session
        yield client


def seed(session, *messages):
    session.app.update_state(session.config, {"messages": list(messages)})


def sse_events(response):
    """Parse an SSE body into (event, payload) pairs."""
    events = []
    for frame in response.text.split("\n\n"):
        if not frame.strip():
            continue
        event, data = "message", ""
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        events.append((event, json.loads(data)))
    return events


# --- pages ------------------------------------------------------------------


def test_index_and_assets_are_served(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


# --- state ------------------------------------------------------------------


def test_state_is_empty_for_a_new_thread(client):
    body = client.get("/api/state").json()
    assert body["thread_id"] == "test"
    assert body["messages"] == []
    assert body["pending_tool_calls"] == []
    assert body["overseer"] == {
        "active": False,
        "goal": "",
        "step": 0,
        "last_instruction": None,
    }


def test_state_serializes_every_message_kind(client):
    seed(
        client.session,
        HumanMessage(content="read the file", id="h1"),
        AIMessage(
            content="",
            id="a1",
            tool_calls=[{"id": "c1", "name": "run_bash", "args": {"cmd": "ls"}}],
        ),
        ToolMessage(content="main.py", tool_call_id="c1", name="run_bash", id="t1"),
        AIMessage(content="There is one file.", id="a2"),
    )
    messages = client.get("/api/state").json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[1]["tool_calls"][0]["name"] == "run_bash"
    assert messages[2]["content"] == "main.py"


def test_empty_ai_messages_are_dropped(client):
    seed(client.session, AIMessage(content="", id="a1"))
    assert client.get("/api/state").json()["messages"] == []


# --- chat -------------------------------------------------------------------


def test_chat_rejects_an_empty_prompt(client):
    assert client.post("/api/chat", json={"prompt": "   "}).status_code == 400


def test_chat_is_unavailable_without_a_model(client, monkeypatch):
    monkeypatch.setattr(client.session, "is_ready", lambda: False)
    assert client.post("/api/chat", json={"prompt": "hi"}).status_code == 503


def test_chat_streams_tokens_then_state(client, monkeypatch):
    monkeypatch.setattr(client.session, "is_ready", lambda: True)
    monkeypatch.setattr(client.session, "stream", lambda prompt: iter(["Hel", "lo"]))
    events = sse_events(client.post("/api/chat", json={"prompt": "hi"}))
    assert [e for e, _ in events] == ["token", "token", "state"]
    assert "".join(d["text"] for e, d in events if e == "token") == "Hello"


def test_a_failing_turn_reports_the_error_and_still_sends_state(client, monkeypatch):
    def boom(prompt):
        raise RuntimeError("model exploded")
        yield  # pragma: no cover - makes this a generator

    monkeypatch.setattr(client.session, "is_ready", lambda: True)
    monkeypatch.setattr(client.session, "stream", boom)
    events = sse_events(client.post("/api/chat", json={"prompt": "hi"}))
    assert events[0][0] == "error" and "model exploded" in events[0][1]["detail"]
    assert events[-1][0] == "state"


def test_the_lock_is_released_after_a_turn(client, monkeypatch):
    monkeypatch.setattr(client.session, "is_ready", lambda: True)
    monkeypatch.setattr(client.session, "stream", lambda prompt: iter(["ok"]))
    client.post("/api/chat", json={"prompt": "hi"})
    assert client.post("/api/threads/new").status_code == 200


# --- tool approval ----------------------------------------------------------


def test_approve_and_feedback_need_a_pending_call(client):
    assert client.post("/api/tools/approve").status_code == 400
    assert client.post("/api/tools/feedback", json={"feedback": "no"}).status_code == 400


def test_approve_resumes_the_graph(client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        client.session,
        "pending_tool_calls",
        lambda: [PendingToolCall(id="c1", name="run_bash", args={})],
    )
    monkeypatch.setattr(client.session, "resume", lambda: calls.append("resumed"))
    events = sse_events(client.post("/api/tools/approve"))
    assert calls == ["resumed"]
    assert events[-1][0] == "state"


def test_deny_drops_the_turn(client):
    seed(
        client.session,
        HumanMessage(content="delete everything", id="h1"),
        AIMessage(content="sure", id="a1"),
    )
    body = client.post("/api/tools/deny").json()
    assert body["messages"] == []


# --- history ----------------------------------------------------------------


def test_clear_history_empties_the_thread(client):
    seed(client.session, HumanMessage(content="hi", id="h1"))
    assert client.post("/api/history/clear").json()["messages"] == []


def test_undo_last_turn(client):
    seed(
        client.session,
        HumanMessage(content="first", id="h1"),
        AIMessage(content="ok", id="a1"),
        HumanMessage(content="second", id="h2"),
    )
    messages = client.post("/api/history/undo_last").json()["messages"]
    assert [m["content"] for m in messages] == ["first", "ok"]


def test_delete_message(client):
    seed(
        client.session,
        HumanMessage(content="first", id="h1"),
        AIMessage(content="ok", id="a1"),
    )
    body = client.post("/api/history/delete_message", json={"message_id": "a1"})
    assert [m["content"] for m in body.json()["messages"]] == ["first"]


# --- threads ----------------------------------------------------------------


def test_threads_listing_includes_the_current_thread(client):
    body = client.get("/api/threads").json()
    assert body["current"] == "test"
    assert any(t["thread_id"] == "test" for t in body["threads"])


def test_switch_and_new_thread(client):
    assert client.post("/api/threads/switch", json={"thread_id": "other"}).json()[
        "thread_id"
    ] == "other"
    assert client.post("/api/threads/new").json()["thread_id"] != "other"


def test_deleting_the_current_thread_moves_to_a_fresh_one(client, monkeypatch):
    monkeypatch.setattr(client.session, "delete_thread", lambda thread_id: None)
    assert client.post("/api/threads/delete", json={"thread_id": "test"}).json()[
        "thread_id"
    ] != "test"


def test_rename_rejects_an_empty_name(client):
    body = {"old_id": "test", "new_id": "  "}
    assert client.post("/api/threads/rename", json=body).status_code == 400


# --- overseer ---------------------------------------------------------------


def test_overseer_start_stop(client):
    body = client.post("/api/overseer/start", json={"goal": "ship it"}).json()
    assert body["overseer"]["active"] is True and body["overseer"]["goal"] == "ship it"
    assert client.post("/api/overseer/stop").json()["overseer"]["active"] is False


def test_overseer_start_rejects_an_empty_goal(client):
    assert client.post("/api/overseer/start", json={"goal": " "}).status_code == 400


def test_overseer_step_requires_an_active_run(client):
    assert client.post("/api/overseer/step").status_code == 400


def test_overseer_step_returns_the_next_instruction(client, monkeypatch):
    monkeypatch.setattr(api._overseer, "advance", lambda messages: "run the tests")
    client.post("/api/overseer/start", json={"goal": "ship it"})
    body = client.post("/api/overseer/step").json()
    assert body["instruction"] == "run the tests"


def test_a_failing_overseer_stops_the_run(client, monkeypatch):
    def boom(messages):
        raise RuntimeError("no API key")

    client.post("/api/overseer/start", json={"goal": "ship it"})
    monkeypatch.setattr(api._overseer, "advance", boom)
    assert client.post("/api/overseer/step").status_code == 502
    assert client.get("/api/state").json()["overseer"]["active"] is False


def test_editing_history_resets_the_overseer(client):
    client.post("/api/overseer/start", json={"goal": "ship it"})
    assert client.post("/api/history/clear").json()["overseer"]["active"] is False


# --- files and notes --------------------------------------------------------


def test_files_listing_and_read(client):
    files = client.get("/api/files").json()["files"]
    assert any(f.endswith("main.py") for f in files)
    body = client.get("/api/files/read", params={"path": "main.py"}).json()
    assert "streamlit" in body["content"].lower()


def test_reading_outside_the_project_is_refused(client):
    res = client.get("/api/files/read", params={"path": "../../etc/passwd"})
    assert res.status_code == 400


def test_writing_outside_the_project_is_refused(client):
    res = client.post(
        "/api/files/write", json={"path": "../escape.txt", "content": "nope"}
    )
    assert res.status_code == 400


def test_missing_files_are_404(client):
    res = client.get("/api/files/read", params={"path": "no_such_file.py"})
    assert res.status_code == 404


def test_file_round_trip(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "note.txt").write_text("before")
    client.post("/api/files/write", json={"path": "note.txt", "content": "after"})
    assert (tmp_path / "note.txt").read_text() == "after"
    assert client.get("/api/files/read", params={"path": "note.txt"}).json()[
        "content"
    ] == "after"


def test_notes_round_trip(client, tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_DATA_ROOT", str(tmp_path))
    client.post("/api/notes", json={"content": "remember this"})
    assert client.get("/api/notes").json()["content"] == "remember this"
