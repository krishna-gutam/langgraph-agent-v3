"""
frontends/streamlit_app.py
--------------------------
The Streamlit frontend. It renders widgets and calls into
`backend.AgentSession` / `backend.Overseer`; it holds no agent logic of its
own, so swapping it for a TUI means rewriting this file only.

Run it with:  streamlit run main.py

Everything lives inside `main()` rather than at module level. Streamlit
re-executes its entrypoint on every interaction, and `main.py` reaches this
code through an import - which Python caches, so module-level rendering would
draw the page once and then render nothing at all on the next click.
"""

import os
import uuid

import streamlit as st
from streamlit_ace import st_ace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend import AgentSession, Overseer, sanitize_content
from backend import workspace


# --- SESSION WIRING ---------------------------------------------------------


@st.cache_resource
def get_session(cwd: str) -> AgentSession:
    """One AgentSession per workspace directory, kept across reruns."""
    return AgentSession(cwd=cwd)


def get_overseer() -> Overseer:
    if "overseer" not in st.session_state:
        st.session_state.overseer = Overseer()
    return st.session_state.overseer


def switch_workspace_environment(overseer: Overseer) -> None:
    """Re-bind the session after chdir into another project."""
    get_session.clear()
    overseer.reset()
    st.rerun()


# --- SIDEBAR ----------------------------------------------------------------


def render_sidebar(session: AgentSession, overseer: Overseer) -> bool:
    """Draw the sidebar. Returns whether tools should be auto-approved."""
    with st.sidebar:
        with st.container(border=True):
            all_threads = session.list_threads()
            current_idx = all_threads.index(session.thread_id)

            selected_thread = st.selectbox(
                "Switch Conversation",
                all_threads,
                index=current_idx,
                format_func=lambda x: x[:8],
            )

            if selected_thread != session.thread_id:
                overseer.reset()
                session.switch_thread(selected_thread)
                st.rerun()

            if st.button("➕ New Conversation", key="new_conv_btn"):
                st.session_state.show_new_thread_input = True

            if st.session_state.get("show_new_thread_input", False):
                custom_id = st.text_input("Enter Thread ID:", key="custom_thread_id_input")
                col1, col2 = st.columns(2)
                if col1.button("Create"):
                    st.session_state.show_new_thread_input = False
                    overseer.reset()
                    session.new_thread(custom_id or None)
                    st.rerun()
                if col2.button("Cancel"):
                    st.session_state.show_new_thread_input = False
                    st.rerun()

        with st.container(border=True):
            project_opts = ["Current Directory"] + workspace.load_recent_projects()

            selected_proj = st.selectbox("Switch Workspace", project_opts)

            if st.button("➕ Create New Project", key="new_proj_btn"):
                st.session_state.show_new_project_input = True

            if st.session_state.get("show_new_project_input", False):
                new_path = st.text_input(
                    "Enter absolute path for new project:", key="new_proj_path_input"
                )
                col1, col2 = st.columns(2)
                if col1.button("Create Project"):
                    if new_path and os.path.isdir(os.path.dirname(new_path)):
                        os.makedirs(new_path, exist_ok=True)
                        st.session_state.show_new_project_input = False
                        os.chdir(new_path)
                        workspace.save_recent_project(new_path)
                        switch_workspace_environment(overseer)
                    else:
                        st.error("Invalid path provided.")
                if col2.button("Cancel Project"):
                    st.session_state.show_new_project_input = False
                    st.rerun()

            if selected_proj != "Current Directory":
                if os.path.exists(selected_proj) and os.getcwd() != selected_proj:
                    os.chdir(selected_proj)
                    workspace.save_recent_project(selected_proj)
                    switch_workspace_environment(overseer)

            st.caption(f"**Active:** `{os.getcwd()}`")

        with st.container(border=True):
            st.metric(label="Conversation Tokens", value=session.token_count)

            auto_approve = st.checkbox("Auto-Approve Tools", value=False)

            if st.button("⏮️ Undo First Turn", use_container_width=True):
                overseer.reset()
                if session.undo_first_turn():
                    st.rerun()

            if st.button("↩️ Undo Last Turn", use_container_width=True):
                overseer.reset()
                if session.undo_last_turn():
                    st.rerun()

            if st.button("🗑️ Clear Chat History", use_container_width=True):
                # Stop autopilot FIRST, or it refills the transcript on the next rerun.
                overseer.reset()
                session.clear_history()
                st.rerun()

        render_overseer_panel(overseer)

        with st.container(border=True):
            sidebar_notes = st.text_area(
                "Quick Notes:", value=workspace.read_notes(), height=200, key="sidebar_notes"
            )
            if st.button("Save Quick Notes"):
                workspace.write_notes(sidebar_notes)
                st.success("Notes saved!")

    return auto_approve


def render_overseer_panel(overseer: Overseer) -> None:
    with st.container(border=True):
        st.markdown("**🎯 Overseer**")

        if not overseer.active:
            goal_text = st.text_area(
                "Goal",
                value=overseer.goal,
                height=110,
                placeholder="e.g. Add a --dry-run flag to cli.py and update the README",
                key="overseer_goal_input",
            )
            if st.button("▶️ Start Overseer", use_container_width=True, type="primary"):
                if goal_text.strip():
                    overseer.start(goal_text)
                    st.rerun()
                else:
                    st.warning("Give it a goal first.")
            st.caption(
                "Tip: turn on Auto-Approve Tools above, or it will stop at every tool call."
            )
        else:
            st.caption(f"**Goal:** {overseer.goal}")
            st.caption(f"♾️ Step {overseer.step} - runs until you stop it")
            if overseer.last_instruction:
                st.caption(f"_Last: {overseer.last_instruction[:120]}_")
            if st.button("⏹️ Stop Overseer", use_container_width=True, type="primary"):
                overseer.stop()
                st.rerun()


# --- TABS -------------------------------------------------------------------


def render_history_tab(session: AgentSession) -> None:
    st.subheader("Conversation Threads")

    for tid in session.list_threads():
        col1, col2, col3, col4 = st.columns([0.85, 0.05, 0.05, 0.05])
        summary = session.thread_summary(tid)

        with col1:
            with st.expander(f"Thread: {tid}"):
                st.write(f"**Last Human:** {summary['last_human'] or 'No human message'}")
                st.write(f"**Last AI:** {summary['last_ai'] or 'No AI message'}")

        with col2:
            if st.button("D", key=f"del_thread_{tid}"):
                session.delete_thread(tid)
                st.rerun()
        with col3:
            new_id = st.text_input(
                "New ID", key=f"rename_input_{tid}", label_visibility="collapsed"
            )
        with col4:
            if st.button("R", key=f"rename_btn_{tid}"):
                if new_id and new_id != tid:
                    session.rename_thread(tid, new_id)
                    st.rerun()


def render_logs_tab(session: AgentSession, overseer: Overseer, graph_messages) -> None:
    st.subheader("Full Message History")
    for i, msg in enumerate(graph_messages):
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            label = f"{i}: {type(msg).__name__}"
            if getattr(msg, "tool_calls", None):
                label += f" ({', '.join(call['name'] for call in msg.tool_calls)})"
            with st.expander(label):
                st.write(msg)
        with col2:
            if st.button("🗑️", key=f"del_{i}"):
                if msg.id:
                    # Hand-editing the transcript is a human intervention, so
                    # stop autopilot rather than letting it react to a history
                    # the user is still in the middle of changing.
                    overseer.reset()
                    session.delete_message(msg.id)
                    st.rerun()


def render_editor_tab() -> None:
    st.subheader("File Editor")

    edit_path = st.selectbox("Select a file to edit:", workspace.list_project_files())

    if st.button("Load File"):
        if edit_path and os.path.exists(edit_path):
            st.session_state.edit_content = workspace.read_file(edit_path)
            # Unique key forces the Ace editor to remount with the new text
            st.session_state.editor_key = str(uuid.uuid4())
        else:
            st.error("File not found.")

    if "edit_content" in st.session_state:
        if "editor_key" not in st.session_state:
            st.session_state.editor_key = "ace_editor_initial"

        new_content = st_ace(
            value=st.session_state.edit_content,
            language="python",
            theme="monokai",
            key=st.session_state.editor_key,
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("💾 Save Changes", use_container_width=True):
                result = workspace.write_file(edit_path, new_content)
                if result.startswith("Error"):
                    st.error(result)
                else:
                    st.success(result)
                    st.session_state.edit_content = new_content

        with col2:
            if st.button("🔄 Reset Unsaved Changes", use_container_width=True):
                if os.path.exists(edit_path):
                    st.session_state.edit_content = workspace.read_file(edit_path)
                    st.session_state.editor_key = str(uuid.uuid4())
                    st.rerun()


# --- CHAT -------------------------------------------------------------------


def render_transcript(graph_messages) -> None:
    """Chat history, rendered straight from LangGraph memory."""
    for msg in graph_messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(sanitize_content(msg.content))
        elif isinstance(msg, AIMessage):
            content = sanitize_content(msg.content)
            # Only render if there is actual text (ignores silent tool-calls)
            if content.strip():
                with st.chat_message("assistant"):
                    st.markdown(content)
        elif isinstance(msg, ToolMessage):
            with st.chat_message("tool", avatar="🔧"):
                with st.expander(f"Result from {msg.name or 'Tool'}", expanded=False):
                    st.code(msg.content, language="markdown")


def render_tool_approval(session: AgentSession, pending_calls, auto_approve: bool) -> None:
    """The approval gate shown whenever the graph paused before the tools node."""
    with st.chat_message("assistant"):
        st.warning("⚠️ **The Agent has requested to execute the following tool(s):**")
        for call in pending_calls:
            with st.expander(f"Tool Call: {call.name}", expanded=True):
                for key, value in call.display_args.items():
                    st.markdown(f"**{key}:**")
                    st.code(str(value), language="python")
                if call.justification:
                    st.markdown("**Justification:**")
                    st.code(call.justification, language="python")

        if auto_approve:
            st.info("Auto-approving because the checkbox is ticked...")
            session.approve_tools()
            st.rerun()

        col1, col2, col3 = st.columns([0.4, 0.3, 0.3])

        if col1.button("✅ Approve Action"):
            with st.status("Executing tools...", expanded=True) as status:
                session.approve_tools()
                status.update(label="Action complete!", state="complete", expanded=False)
            st.rerun()

        if col2.button("❌ Deny Action"):
            session.deny_tools()
            st.error("Action denied and forgotten from agent memory.")
            st.rerun()

        with col3:
            with st.popover("💬 Provide Feedback"):
                feedback_text = st.text_area("Tell the agent what to change:")

                if st.button("Submit Feedback"):
                    if feedback_text.strip():
                        session.send_tool_feedback(feedback_text)
                        # Resume outside the popover, on the next rerun
                        st.session_state.resume_agent = True
                        st.rerun()
                    else:
                        st.warning("Please enter some feedback.")


def run_prompt(session: AgentSession, prompt: str) -> None:
    """Stream one turn into the chat feed, then rerun to show the settled state."""
    safe_prompt = sanitize_content(prompt)

    with st.chat_message("user"):
        st.markdown(safe_prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        stream_failed = False

        try:
            for chunk in session.stream(safe_prompt):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            if full_response:
                message_placeholder.markdown(full_response)
        except Exception as e:
            message_placeholder.error(f"Error: {e}")
            stream_failed = True

    # Outside the try: st.rerun() raises, and catching it there would swallow
    # the rerun instead of redrawing the page.
    if not stream_failed:
        st.rerun()


# --- ENTRY POINT ------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="----------", layout="wide")

    session = get_session(os.getcwd())
    overseer = get_overseer()

    auto_approve = render_sidebar(session, overseer)

    graph_messages = session.messages
    pending_calls = session.pending_tool_calls()

    tab_chat, tab_edit, tab_logs, tab_history = st.tabs(
        [
            "💬 Chat Interface",
            "📝 Editor",
            "📜 Message Logs",
            "🕒 Manage History",
        ]
    )

    with tab_history:
        render_history_tab(session)

    with tab_logs:
        render_logs_tab(session, overseer, graph_messages)

    with tab_edit:
        render_editor_tab()

    with tab_chat:
        render_transcript(graph_messages)

        if pending_calls:
            render_tool_approval(session, pending_calls, auto_approve)

        # Resume a graph that was answered with feedback on the previous run.
        if st.session_state.get("resume_agent", False):
            st.session_state.resume_agent = False
            with st.chat_message("assistant"):
                with st.spinner("Processing your feedback..."):
                    session.resume()
            st.rerun()

        # Overseer - one instruction per rerun. Only the Stop button ends this.
        if overseer.active and not overseer.pending_prompt:
            if pending_calls:
                st.info("⏸️ Overseer is waiting for you to approve the tool call above.")
            else:
                try:
                    with st.spinner("Overseer is writing the next instruction..."):
                        overseer.advance(graph_messages)
                except RuntimeError as e:
                    overseer.stop()
                    st.error(f"🎯 Overseer stopped: {e}")
                    st.stop()
                st.rerun()

    # --- NEW USER INPUT ---

    if not session.is_ready():
        with tab_chat:
            st.warning(
                "⚠️ **API Key Required:** Set GOOGLE_API_KEY and MODEL_ID in your .env to start coding."
            )
        st.chat_input("Configure your API key to chat...", disabled=True)
        return

    typed_prompt = st.chat_input("What would you like to do?")

    # An Overseer instruction is consumed here exactly like typed input.
    # If you type during a run, your message wins for this turn - the
    # Overseer just re-reads the transcript on its next cycle.
    prompt = typed_prompt or overseer.take_pending()

    if prompt:
        with tab_chat:
            run_prompt(session, prompt)


if __name__ == "__main__":
    main()
