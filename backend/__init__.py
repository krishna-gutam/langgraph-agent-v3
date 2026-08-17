"""
backend
-------
The UI-agnostic half of the agent: the LangGraph graph, the tools, workspace
storage, the Overseer, and the `AgentSession` facade that ties them together.

Nothing in this package imports a UI framework, so any frontend - Streamlit,
a TUI, a plain terminal, an HTTP server - can drive it:

    from backend import AgentSession

    session = AgentSession()
    for chunk in session.stream("list the files in tools/"):
        print(chunk, end="")
    for call in session.pending_tool_calls():
        print(call.name, call.args)
    session.approve_tools()
"""

from backend.graph import AgentState, create_graph, get_llm, sanitize_content
from backend.session import AgentSession, Overseer, PendingToolCall, estimate_tokens

__all__ = [
    "AgentSession",
    "AgentState",
    "Overseer",
    "PendingToolCall",
    "create_graph",
    "estimate_tokens",
    "get_llm",
    "sanitize_content",
]
