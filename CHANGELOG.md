# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed
- **Backend / frontend separation**: agent logic now lives in a UI-agnostic
  `backend/` package (`graph.py`, `overseer.py`, `workspace.py`, `session.py`),
  and the UIs live in `frontends/` (`streamlit_app.py`, `cli.py`). `main.py` and
  `cli.py` are thin launchers, so existing commands are unchanged.
- **`AgentSession`**: single entry point for any frontend - streaming, pending
  tool calls, approve/deny/feedback, undo, clear, thread management, token count.
- **`Overseer`**: autopilot state is now a plain object instead of Streamlit
  session state, so any frontend can drive it.
- **CLI**: gained `/clear`, `/undo`, `/threads`, `/new`, `/tokens`, tool
  feedback, `--persist`, `--thread`, `--auto-approve`, and `--goal` autopilot.

### Removed
- `utils/helpers.py` - dead code that imported Streamlit inside a utility module
  and duplicated `sanitize_content`.

## [1.1.0] - 2025-03-09

### Added
- **Comprehensive Test Suite**:
  - `tests/test_agent_core.py`: Verifies LangGraph compilation (`create_graph`) and initial agent state initialization with checkpointer.
  - `tests/test_tools.py`: Verifies the tool registry successfully loads all expected tools (`run_bash`, `apply_patch`, `web_search`, `web_fetch`).
  - `tests/test_cli.py`: Covers command-line interface execution (`run_cli`), handling user messaging, immediate exit, tool approval (`y`), and tool denial (`n`).
- **Git Tracking**: Staged all new test files and changelog into git version control.
