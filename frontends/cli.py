"""
frontends/cli.py
----------------
A terminal frontend. It owns printing and prompting; every decision about the
agent itself lives in `backend.AgentSession`.

Use it as a reference when writing another frontend (a TUI, a bot, an HTTP
server): the shape below - stream, check for pending tool calls, approve or
deny - is the whole contract.
"""

import argparse

from langgraph.checkpoint.memory import MemorySaver

from backend import AgentSession, Overseer

BANNER = """--- Coding Agent CLI ---
Type 'exit' to quit. Commands: /clear  /undo  /threads  /new [id]  /tokens"""


def _render_tool_calls(calls) -> None:
    print("\n[!] The agent wants to run:")
    for call in calls:
        print(f"  - {call.name}")
        for key, value in call.display_args.items():
            print(f"      {key}: {value}")
        if call.justification:
            print(f"      justification: {call.justification}")


def handle_pending_tools(session: AgentSession, auto_approve: bool = False) -> None:
    """Approve, deny, or send feedback for whatever the graph paused on."""
    while True:
        calls = session.pending_tool_calls()
        if not calls:
            return

        _render_tool_calls(calls)

        if auto_approve:
            choice = "y"
            print("Auto-approving.")
        else:
            choice = input("Approve? (y = run / n = deny / f = feedback) > ").strip().lower()

        if choice == "y":
            print("Executing...")
            session.approve_tools()
            last = session.messages[-1]
            print("Result:", getattr(last, "content", ""))
        elif choice == "f":
            feedback = input("Feedback for the agent: ").strip()
            if not feedback:
                continue
            session.send_tool_feedback(feedback)
            print("Agent: ", end="", flush=True)
            session.resume()
            print(getattr(session.messages[-1], "content", ""))
        else:
            session.deny_tools()
            print("Action denied and forgotten from agent memory.")
            return


def handle_command(session: AgentSession, overseer: Overseer, line: str) -> bool:
    """Run a slash command. Returns True if the line was a command."""
    if not line.startswith("/"):
        return False

    command, _, arg = line[1:].partition(" ")
    arg = arg.strip()

    if command == "clear":
        overseer.reset()
        session.clear_history()
        print("History cleared.")
    elif command == "undo":
        print("Undone." if session.undo_last_turn() else "Nothing to undo.")
        overseer.reset()
    elif command == "threads":
        for tid in session.list_threads():
            marker = "*" if tid == session.thread_id else " "
            print(f" {marker} {tid}")
    elif command == "new":
        overseer.reset()
        print(f"Switched to thread {session.new_thread(arg or None)}")
    elif command == "tokens":
        print(f"~{session.token_count} tokens")
    else:
        print(f"Unknown command: /{command}")
    return True


def run_turn(session: AgentSession, prompt: str, auto_approve: bool = False) -> None:
    """One full exchange: stream the reply, then settle any tool call."""
    print("Agent: ", end="", flush=True)
    for chunk in session.stream(prompt):
        print(chunk, end="", flush=True)
    print()
    handle_pending_tools(session, auto_approve=auto_approve)


def run_overseer(session: AgentSession, overseer: Overseer, goal: str, steps: int) -> None:
    """Autopilot: the Overseer writes each instruction, tools auto-approve."""
    overseer.start(goal)
    print(f"🎯 Overseer running toward: {goal}\n")
    for _ in range(steps):
        try:
            instruction = overseer.advance(session.messages)
        except RuntimeError as e:
            print(f"🎯 Overseer stopped: {e}")
            return
        print(f"\n[Overseer step {overseer.step}] {instruction}\n")
        run_turn(session, instruction, auto_approve=True)
    print("\n🎯 Step limit reached.")


def run_cli(session: AgentSession | None = None, auto_approve: bool = False) -> None:
    """Interactive loop. Uses a transient in-memory session unless given one."""
    if session is None:
        session = AgentSession(thread_id="cli_session", checkpointer=MemorySaver())
    overseer = Overseer()

    print(BANNER)
    if not session.is_ready():
        print("⚠️  No LLM configured - set GOOGLE_API_KEY and MODEL_ID in .env")

    while True:
        try:
            user_input = input("\nUser: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.strip().lower() in ("exit", "quit"):
            break
        if not user_input.strip():
            continue
        if handle_command(session, overseer, user_input.strip()):
            continue

        try:
            run_turn(session, user_input, auto_approve=auto_approve)
        except Exception as e:
            print(f"\nError: {e}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Coding agent - terminal frontend")
    parser.add_argument(
        "--persist",
        action="store_true",
        help="use the workspace SQLite history instead of a throwaway session",
    )
    parser.add_argument("--thread", help="thread ID to resume (implies --persist)")
    parser.add_argument("--auto-approve", action="store_true", help="run tools without asking")
    parser.add_argument("--goal", help="run the Overseer toward this goal instead of chatting")
    parser.add_argument("--steps", type=int, default=10, help="Overseer step limit (default 10)")
    args = parser.parse_args(argv)

    if args.persist or args.thread:
        session = AgentSession(thread_id=args.thread)
    else:
        session = AgentSession(thread_id="cli_session", checkpointer=MemorySaver())

    if args.goal:
        run_overseer(session, Overseer(), args.goal, args.steps)
    else:
        run_cli(session, auto_approve=args.auto_approve)


if __name__ == "__main__":
    main()
