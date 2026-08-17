# LangGraph Coding Agent

A powerful, modular, and autonomous AI coding assistant built with **LangGraph**, **Streamlit**, and **Google Gemini**. This agent is designed to handle complex coding tasks by planning, executing, and verifying code changes, with a unique "Overseer" autopilot mode.

## 🚀 Key Features

*   **Interactive Streamlit UI**: A full-featured web interface for chatting, file editing, managing conversation history, and monitoring agent activity.
*   **Web UI (FastAPI)**: A dependency-free HTML/CSS/JS front end served by FastAPI, with streaming replies, tool approval, conversation history, and a file editor.
*   **CLI Interface**: A lightweight, terminal-based interface for quick tasks.
*   **Overseer (Autopilot) Mode**: A meta-agent that acts as a proxy for the user. It plans, steers, and verifies the coding agent's work toward a specific goal, pausing only when human intervention is required.
*   **Sophisticated Tooling**:
    *   **`apply_patch`**: A robust, fuzzy-matching file editor that handles indentation and whitespace variations gracefully using a 9-tier strategy.
    *   **`run_bash`**: A secure, environment-aware bash executor that automatically manages a project-specific `.venv`.
    *   **`web_search` & `web_fetch`**: Integrated research capabilities using Tavily and BeautifulSoup.
*   **Persistent Memory**: Uses SQLite to maintain conversation state across sessions.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**:
    Create a `.env` file in the root directory and add your API keys:
    ```env
    GOOGLE_API_KEY=your_gemini_api_key
    MODEL_ID=gemini-2.0-flash # or your preferred model
    TAVILY_API_KEY=your_tavily_api_key
    AGENT_DATA_ROOT=./data # Directory for workspace and history
    ```

## 🏛️ Project Layout

The agent is split into a UI-agnostic backend and interchangeable frontends, so
you can build a new UI (TUI, HTTP API, chat bot) without touching agent logic.

```
backend/            no UI framework is imported here
  graph.py          the LangGraph graph, the LLM, sanitize_content
  overseer.py       the meta-agent that writes the next instruction
  workspace.py      data dirs, SQLite checkpoints, threads, projects, notes
  session.py        AgentSession + Overseer - the API a frontend calls
frontends/
  streamlit_app.py  the Streamlit web UI
  cli.py            the terminal UI
  web/              the FastAPI web UI
    api.py          HTTP + SSE endpoints over AgentSession
    static/         index.html, styles.css, app.js (no build step, no CDN)
main.py             launcher: streamlit run main.py
cli.py              launcher: python cli.py
```

### Writing your own frontend

Everything a UI needs is on `AgentSession`:

```python
from backend import AgentSession, Overseer

session = AgentSession()                      # persistent, per-workspace history
for chunk in session.stream("refactor cli.py"):
    print(chunk, end="")                      # token-by-token reply

for call in session.pending_tool_calls():     # empty unless the graph paused
    print(call.name, call.display_args, call.justification)
session.approve_tools()                       # or .deny_tools() / .send_tool_feedback("...")

session.undo_last_turn(); session.clear_history()
session.list_threads(); session.switch_thread("my-thread")
print(session.token_count, session.is_ready())

overseer = Overseer()                         # autopilot; the frontend owns the loop
overseer.start("add a --dry-run flag")
session.send(overseer.advance(session.messages))
```

Pass `checkpointer=MemorySaver()` for a throwaway session (that is what the CLI
does by default), or omit it to use the workspace's SQLite history.

## 📖 Usage

### Streamlit UI
Launch the web interface to start coding:
```bash
streamlit run main.py
```
*   **Chat**: Interact with the agent directly.
*   **Editor**: Load, edit, and save files directly from the browser.
*   **Overseer**: Set a goal and let the agent work autonomously.

### FastAPI web UI
A browser UI with no build step - plain HTML, CSS and JavaScript served by FastAPI:
```bash
python -m frontends.web            # http://127.0.0.1:8000
uvicorn frontends.web:app --reload # same thing, with reload
```
`WEB_HOST`, `WEB_PORT` and `WEB_RELOAD` override the defaults. The UI covers
chat (streamed over Server-Sent Events), tool approval / denial / replying to a
tool call, conversation switching and deletion, undo and clear, notes, the
Overseer, and a file browser and editor.

The Overseer panel mirrors `python cli.py --goal ... --steps N`: give it a goal,
a step limit, and whether tools are auto-approved. With auto-approve on it runs
unattended (the CLI's behaviour); with it off the run parks at each tool call
and resumes when you approve. Denying a call stops the run, since the turn it
came from is discarded.

The JSON API underneath is usable on its own: `GET /api/state`,
`POST /api/chat`, `POST /api/tools/{approve,deny,feedback}`,
`GET|POST /api/threads/*`, `POST /api/history/*`, `POST /api/overseer/*`,
`GET|POST /api/files` and `/api/notes`.

> It serves a local coding agent with no authentication and binds to localhost.
> Do not expose it to a network you do not control.

### CLI
For quick, terminal-based interactions:
```bash
python cli.py                       # transient session
python cli.py --persist             # reuse the workspace's saved history
python cli.py --thread my-thread    # resume a specific conversation
python cli.py --auto-approve        # run tools without prompting
python cli.py --goal "add a --dry-run flag to cli.py" --steps 5   # Overseer autopilot
```
In-chat commands: `/clear`, `/undo`, `/threads`, `/new [id]`, `/tokens`.

## 🛠️ Available Tools

| Tool | Description |
| :--- | :--- |
| `apply_patch` | Edits files using a 9-tier fuzzy matching strategy to ensure accurate code replacement. |
| `run_bash` | Executes shell commands in a sandboxed, auto-managed virtual environment. |
| `web_search` | Performs advanced web searches via Tavily. |
| `web_fetch` | Scrapes and cleans text content from URLs. |

## 🎯 The Overseer Architecture

The **Overseer** is a meta-agent designed to steer the coding agent without being part of the core LangGraph state.

*   **How it works**: It observes the conversation transcript and generates the next `HumanMessage`. This keeps the agent's history clean and makes the feature modular.
*   **Decision Loop**: It evaluates the current state against the user's goal and returns a JSON decision:
    *   `continue`: Provides the next instruction.
    *   `goal_reached`: Signals completion.
    *   `needs_human`: Escalates to the user for critical decisions (e.g., destructive actions, ambiguous goals).
*   **Safety**: It is programmed to avoid repeating instructions and to verify results before declaring victory.



## 🧪 Running Tests

To run the test suite and verify core functionality, tool registration, and CLI behavior:
```bash
PYTHONPATH=. pytest
```
