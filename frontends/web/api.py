"""
frontends/web/api.py
--------------------
A FastAPI frontend for the coding agent.

Same rules as the other frontends: this module drives `AgentSession` and owns
nothing the backend already owns. It adds the two things HTTP needs and the
backend does not have - JSON shapes for LangChain messages, and a lock, since
one `AgentSession` (and the SQLite connection under it) is shared by every
request in the process.

Run it with:

    uvicorn frontends.web:app --reload
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from backend import AgentSession, Overseer, sanitize_content, workspace

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="LangGraph Coding Agent", version="1.0")

# One session per process. The graph streams synchronously and SqliteSaver is
# not safe to drive from two turns at once, so every mutating request takes
# this lock; a second turn arriving mid-stream gets 409 rather than a corrupt
# transcript.
_session: AgentSession | None = None
_overseer = Overseer()
_lock = threading.Lock()


def get_session() -> AgentSession:
    global _session
    if _session is None:
        _session = AgentSession()
    return _session


class Busy(HTTPException):
    def __init__(self) -> None:
        super().__init__(status_code=409, detail="The agent is busy with another turn.")


# --- serialization ----------------------------------------------------------


def message_to_dict(msg) -> dict | None:
    """
    One LangChain message as JSON the browser can render. Returns None for
    messages with nothing to show (an AIMessage that is only tool calls is
    represented by its calls, not by an empty bubble).
    """
    text = sanitize_content(msg.content)
    if isinstance(msg, HumanMessage):
        return {"id": msg.id, "role": "user", "content": text}
    if isinstance(msg, ToolMessage):
        return {
            "id": msg.id,
            "role": "tool",
            "name": msg.name,
            "content": text,
        }
    if isinstance(msg, AIMessage):
        calls = [
            {"id": c.get("id"), "name": c.get("name"), "args": c.get("args", {}) or {}}
            for c in getattr(msg, "tool_calls", []) or []
        ]
        if not text and not calls:
            return None
        return {"id": msg.id, "role": "assistant", "content": text, "tool_calls": calls}
    return None


def state_payload() -> dict:
    """Everything the UI needs to redraw itself after any action."""
    session = get_session()
    return {
        "thread_id": session.thread_id,
        "ready": session.is_ready(),
        "cwd": os.getcwd(),
        "token_count": session.token_count,
        "messages": [d for d in (message_to_dict(m) for m in session.messages) if d],
        "pending_tool_calls": [
            {
                "id": call.id,
                "name": call.name,
                "args": call.display_args,
                "justification": call.justification,
            }
            for call in session.pending_tool_calls()
        ],
        "overseer": {
            "active": _overseer.active,
            "goal": _overseer.goal,
            "step": _overseer.step,
            "last_instruction": _overseer.last_instruction,
        },
    }


# --- request bodies ---------------------------------------------------------


class ChatRequest(BaseModel):
    prompt: str


class FeedbackRequest(BaseModel):
    feedback: str


class ThreadRequest(BaseModel):
    thread_id: str


class RenameRequest(BaseModel):
    old_id: str
    new_id: str


class MessageRequest(BaseModel):
    message_id: str


class GoalRequest(BaseModel):
    goal: str


class NotesRequest(BaseModel):
    content: str


class FileWriteRequest(BaseModel):
    path: str
    content: str


# --- state ------------------------------------------------------------------


@app.get("/api/state")
def read_state() -> dict:
    return state_payload()


# --- chat -------------------------------------------------------------------


def _sse(event: str, data: dict) -> str:
    """One SSE frame. The payload is compact JSON so it never spans lines -
    a raw newline inside `data:` would split the frame in two."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    """
    Stream one turn as Server-Sent Events: `token` events as the reply arrives,
    then a final `state` event so the client can pick up tool calls and the new
    token count without a second request.
    """
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty.")
    session = get_session()
    if not session.is_ready():
        raise HTTPException(
            status_code=503,
            detail="The model is not configured. Set GOOGLE_API_KEY and MODEL_ID.",
        )
    if not _lock.acquire(blocking=False):
        raise Busy()

    def events():
        # The lock is held by the request that entered this endpoint and is
        # released here, when the generator finishes or the client disconnects.
        try:
            for chunk in session.stream(prompt):
                yield _sse("token", {"text": chunk})
        except Exception as exc:  # surfaced in the transcript, not swallowed
            yield _sse("error", {"detail": str(exc)})
        finally:
            try:
                yield _sse("state", state_payload())
            finally:
                _lock.release()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/tools/approve")
def approve_tools() -> StreamingResponse:
    """Run the pending tool calls and stream whatever the agent says next."""
    session = get_session()
    if not session.pending_tool_calls():
        raise HTTPException(status_code=400, detail="No tool call is pending.")
    if not _lock.acquire(blocking=False):
        raise Busy()

    def events():
        try:
            session.resume()
        except Exception as exc:
            yield _sse("error", {"detail": str(exc)})
        finally:
            try:
                yield _sse("state", state_payload())
            finally:
                _lock.release()

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/api/tools/deny")
def deny_tools() -> dict:
    with _guard():
        get_session().deny_tools()
    return state_payload()


@app.post("/api/tools/feedback")
def tool_feedback(req: FeedbackRequest) -> StreamingResponse:
    """Answer the pending calls with words instead of running them."""
    session = get_session()
    if not session.pending_tool_calls():
        raise HTTPException(status_code=400, detail="No tool call is pending.")
    if not _lock.acquire(blocking=False):
        raise Busy()

    def events():
        try:
            session.send_tool_feedback(req.feedback)
            session.resume()
        except Exception as exc:
            yield _sse("error", {"detail": str(exc)})
        finally:
            try:
                yield _sse("state", state_payload())
            finally:
                _lock.release()

    return StreamingResponse(events(), media_type="text/event-stream")


class _Guard:
    """`with _guard():` - 409 instead of blocking when a turn is in flight."""

    def __enter__(self):
        if not _lock.acquire(blocking=False):
            raise Busy()
        return self

    def __exit__(self, *exc_info):
        _lock.release()
        return False


def _guard() -> _Guard:
    return _Guard()


# --- history ----------------------------------------------------------------


@app.post("/api/history/undo_last")
def undo_last() -> dict:
    with _guard():
        _overseer.reset()
        get_session().undo_last_turn()
    return state_payload()


@app.post("/api/history/undo_first")
def undo_first() -> dict:
    with _guard():
        _overseer.reset()
        get_session().undo_first_turn()
    return state_payload()


@app.post("/api/history/clear")
def clear_history() -> dict:
    with _guard():
        _overseer.reset()
        get_session().clear_history()
    return state_payload()


@app.post("/api/history/delete_message")
def delete_message(req: MessageRequest) -> dict:
    with _guard():
        _overseer.reset()
        get_session().delete_message(req.message_id)
    return state_payload()


# --- threads ----------------------------------------------------------------


@app.get("/api/threads")
def list_threads() -> dict:
    session = get_session()
    return {
        "current": session.thread_id,
        "threads": [session.thread_summary(t) for t in session.list_threads()],
    }


@app.post("/api/threads/new")
def new_thread() -> dict:
    with _guard():
        _overseer.reset()
        get_session().new_thread()
    return state_payload()


@app.post("/api/threads/switch")
def switch_thread(req: ThreadRequest) -> dict:
    with _guard():
        _overseer.reset()
        get_session().switch_thread(req.thread_id)
    return state_payload()


@app.post("/api/threads/delete")
def delete_thread(req: ThreadRequest) -> dict:
    session = get_session()
    with _guard():
        session.delete_thread(req.thread_id)
        # Deleting the thread you are on leaves the session pointing at a row
        # that no longer exists, so move to a fresh one.
        if req.thread_id == session.thread_id:
            _overseer.reset()
            session.new_thread()
    return state_payload()


@app.post("/api/threads/rename")
def rename_thread(req: RenameRequest) -> dict:
    if not req.new_id.strip():
        raise HTTPException(status_code=400, detail="The new name is empty.")
    with _guard():
        get_session().rename_thread(req.old_id, req.new_id.strip())
    return state_payload()


# --- overseer ---------------------------------------------------------------


@app.post("/api/overseer/start")
def overseer_start(req: GoalRequest) -> dict:
    if not req.goal.strip():
        raise HTTPException(status_code=400, detail="The goal is empty.")
    _overseer.start(req.goal)
    return state_payload()


@app.post("/api/overseer/stop")
def overseer_stop() -> dict:
    _overseer.stop()
    return state_payload()


@app.post("/api/overseer/step")
def overseer_step() -> dict:
    """
    Produce the next autopilot instruction. The client then posts it to
    /api/chat, so one browser-side loop drives the run and can be stopped
    between steps.
    """
    if not _overseer.active:
        raise HTTPException(status_code=400, detail="Autopilot is not running.")
    with _guard():
        try:
            instruction = _overseer.advance(get_session().messages)
        except RuntimeError as exc:
            _overseer.stop()
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"instruction": instruction, "step": _overseer.step}


# --- files and notes --------------------------------------------------------


@app.get("/api/files")
def list_files() -> dict:
    return {"files": sorted(workspace.list_project_files("."))}


def _resolve_in_project(path: str) -> Path:
    """Reject anything outside the working directory - this server is a local
    tool, but a path field is still a path field."""
    root = Path(os.getcwd()).resolve()
    target = (root / path).resolve()
    if target != root and root not in target.parents:
        raise HTTPException(status_code=400, detail="Path is outside the project.")
    return target


@app.get("/api/files/read")
def read_file(path: str) -> dict:
    target = _resolve_in_project(path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="No such file.")
    return {"path": path, "content": workspace.read_file(str(target))}


@app.post("/api/files/write")
def write_file(req: FileWriteRequest) -> dict:
    target = _resolve_in_project(req.path)
    return {"result": workspace.write_file(str(target), req.content)}


@app.get("/api/notes")
def read_notes() -> dict:
    return {"content": workspace.read_notes()}


@app.post("/api/notes")
def write_notes(req: NotesRequest) -> dict:
    workspace.write_notes(req.content)
    return {"content": req.content}


# --- static UI --------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
