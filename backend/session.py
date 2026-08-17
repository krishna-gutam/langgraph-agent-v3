"""
backend/session.py
------------------
The API a frontend talks to.

`AgentSession` owns the compiled graph, the checkpointer and the current
thread. Everything a UI needs - send a message, inspect the pending tool call,
approve/deny it, undo a turn, switch threads - is a method here. Nothing in
this module knows whether the caller is Streamlit, a terminal, or a test.

`Overseer` is the same idea for autopilot: plain state (active/goal/step) plus
a method that returns the next instruction.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Iterator

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from backend import workspace
from backend.graph import create_graph, get_llm, sanitize_content
from backend.overseer import next_instruction


@dataclass
class PendingToolCall:
    """One tool the agent is waiting for permission to run."""

    id: str
    name: str
    args: dict

    @property
    def justification(self) -> str | None:
        return self.args.get("justification")

    @property
    def display_args(self) -> dict:
        """Args minus the justification, which frontends render separately."""
        return {k: v for k, v in self.args.items() if k != "justification"}


def estimate_tokens(messages) -> int:
    """
    Token count for a message list. Prefers real usage_metadata from the most
    recent AIMessage; falls back to characters / 3.675.
    """
    if not messages:
        return 0
    for m in reversed(messages):
        if isinstance(m, AIMessage) and getattr(m, "usage_metadata", None):
            return m.usage_metadata.get("total_tokens", 0)
    total_chars = sum(len(str(m.content)) for m in messages)
    return int(total_chars // 3.675)


class AgentSession:
    """A conversation with the coding agent, bound to one workspace directory."""

    def __init__(self, thread_id: str | None = None, checkpointer=None, cwd=None):
        self.cwd = cwd
        self.checkpointer = (
            checkpointer if checkpointer is not None else workspace.get_checkpointer(cwd)
        )
        self.app = create_graph(checkpointer=self.checkpointer)
        self._persist_thread = checkpointer is None
        self.thread_id = thread_id or self._load_or_create_thread_id()

    # --- threads ---------------------------------------------------------

    def _load_or_create_thread_id(self) -> str:
        saved = workspace.get_saved_thread_id(self.cwd)
        if saved:
            return saved
        new_id = str(uuid.uuid4())
        workspace.save_thread_id(new_id, self.cwd)
        return new_id

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    def switch_thread(self, thread_id: str) -> None:
        """Point this session at another thread. LangGraph creates it lazily."""
        self.thread_id = thread_id
        if self._persist_thread:
            workspace.save_thread_id(thread_id, self.cwd)

    def new_thread(self, thread_id: str | None = None) -> str:
        self.switch_thread(thread_id or str(uuid.uuid4()))
        return self.thread_id

    def list_threads(self) -> list[str]:
        threads = workspace.get_all_thread_ids(self.cwd)
        # A brand new thread has no checkpoint yet, so the DB does not know it.
        if self.thread_id not in threads:
            threads.insert(0, self.thread_id)
        return threads

    def delete_thread(self, thread_id: str) -> None:
        workspace.delete_thread(thread_id, self.cwd)

    def rename_thread(self, old_id: str, new_id: str) -> None:
        workspace.rename_thread(old_id, new_id, self.cwd)
        if self.thread_id == old_id:
            self.switch_thread(new_id)

    def thread_summary(self, thread_id: str) -> dict:
        """Last human and last AI message of a thread, for history listings."""
        state = self.app.get_state({"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", [])
        last_human = last_ai = None
        for msg in reversed(messages):
            if last_human is None and isinstance(msg, HumanMessage):
                last_human = sanitize_content(msg.content)
            if last_ai is None and isinstance(msg, AIMessage):
                last_ai = sanitize_content(msg.content)
            if last_human and last_ai:
                break
        return {
            "thread_id": thread_id,
            "last_human": last_human,
            "last_ai": last_ai,
            "message_count": len(messages),
        }

    # --- state -----------------------------------------------------------

    @property
    def messages(self) -> list:
        return self.app.get_state(self.config).values.get("messages", [])

    @property
    def token_count(self) -> int:
        try:
            return estimate_tokens(self.messages)
        except Exception:
            return 0

    def is_ready(self) -> bool:
        """False when the LLM is not configured (missing API key or model)."""
        return get_llm() is not None

    def pending_tool_calls(self) -> list[PendingToolCall]:
        """Tool calls the graph paused on, or [] when it is not paused."""
        snapshot = self.app.get_state(self.config)
        if snapshot.next != ("tools",):
            return []
        last = snapshot.values["messages"][-1]
        return [
            PendingToolCall(
                id=call.get("id"), name=call.get("name", "Unknown Tool"), args=call.get("args", {}) or {}
            )
            for call in getattr(last, "tool_calls", []) or []
        ]

    @property
    def is_awaiting_approval(self) -> bool:
        return self.app.get_state(self.config).next == ("tools",)

    # --- running ---------------------------------------------------------

    def stream(self, prompt: str) -> Iterator[str]:
        """Send a message and yield assistant text as it arrives."""
        safe_prompt = sanitize_content(prompt)
        for chunk, _metadata in self.app.stream(
            {"messages": [HumanMessage(content=safe_prompt)]},
            config=self.config,
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                yield sanitize_content(chunk.content)

    def send(self, prompt: str) -> str:
        """Blocking version of stream(): returns the whole reply."""
        return "".join(self.stream(prompt))

    def resume(self) -> None:
        """Continue a graph that is paused (approve the tool call, or resume
        after injecting a tool result)."""
        self.app.invoke(None, config=self.config)

    approve_tools = resume

    def deny_tools(self) -> None:
        """Reject the pending call and forget the turn that produced it."""
        messages = self.app.get_state(self.config).values.get("messages", [])
        human_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
        if not human_indices:
            return
        removals = [
            RemoveMessage(id=msg.id) for msg in messages[human_indices[-1]:] if msg.id
        ]
        if removals:
            self.app.update_state(self.config, {"messages": removals})

    def send_tool_feedback(self, feedback: str) -> None:
        """
        Answer the pending tool calls with the user's words instead of running
        them, then let the agent react. The graph resumes from the tools node.
        """
        calls = self.pending_tool_calls()
        if not calls:
            return
        tool_messages = [
            ToolMessage(
                tool_call_id=call.id,
                name=call.name,
                content=f"User interrupted: {feedback}",
            )
            for call in calls
        ]
        self.app.update_state(self.config, {"messages": tool_messages}, as_node="tools")

    # --- history editing -------------------------------------------------

    def _remove(self, messages) -> bool:
        removals = [RemoveMessage(id=m.id) for m in messages if m.id]
        if not removals:
            return False
        self.app.update_state(self.config, {"messages": removals})
        return True

    def delete_message(self, message_id: str) -> None:
        self.app.update_state(self.config, {"messages": [RemoveMessage(id=message_id)]})

    def undo_first_turn(self) -> bool:
        """Drop the first user turn and everything it produced."""
        messages = self.messages
        human_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
        if not human_indices:
            return False
        start = human_indices[0]
        end = human_indices[1] if len(human_indices) > 1 else len(messages)
        return self._remove(messages[start:end])

    def undo_last_turn(self) -> bool:
        """Drop the most recent user turn and everything after it."""
        messages = self.messages
        human_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
        if not human_indices:
            return False
        return self._remove(messages[human_indices[-1]:])

    def clear_history(self) -> None:
        """
        Wipe the whole message list. REMOVE_ALL_MESSAGES rather than slicing -
        slicing left anything before the first HumanMessage (or without an id)
        behind, so "clear" was not always a full clear.
        """
        self.app.update_state(
            self.config, {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
        )


@dataclass
class Overseer:
    """
    Autopilot state plus one method. The frontend owns the loop: while
    `active`, ask for `advance(messages)` and feed the result to the session
    as if the user had typed it.
    """

    active: bool = False
    goal: str = ""
    step: int = 0
    instructions: list[str] = field(default_factory=list)
    pending_prompt: str | None = None

    def start(self, goal: str) -> None:
        self.active = True
        self.goal = goal.strip()
        self.step = 0
        self.instructions = []
        self.pending_prompt = None

    def stop(self) -> None:
        """Stop autopilot and drop any queued instruction."""
        self.active = False
        self.pending_prompt = None

    def reset(self) -> None:
        """
        Full stop, used before any destructive edit to the transcript.

        The instruction log and step count go too: an Overseer that keeps its
        own log while the agent's transcript is empty would send step N+1 to an
        agent with no memory of steps 1..N.
        """
        self.stop()
        self.step = 0
        self.instructions = []

    def advance(self, messages) -> str:
        """
        Produce the next instruction and record it.

        Raises RuntimeError if the Overseer cannot produce one (missing config,
        API failure, empty response) - a genuine failure, not a decision to
        stop. Callers surface it and end the run.
        """
        instruction = next_instruction(
            goal=self.goal, messages=messages, past_instructions=self.instructions
        )
        self.step += 1
        self.instructions.append(instruction)
        self.pending_prompt = instruction
        return instruction

    def take_pending(self) -> str | None:
        prompt, self.pending_prompt = self.pending_prompt, None
        return prompt

    @property
    def last_instruction(self) -> str | None:
        return self.instructions[-1] if self.instructions else None
