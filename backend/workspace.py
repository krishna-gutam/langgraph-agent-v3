"""
backend/workspace.py
--------------------
Everything about *where* the agent stores things: the per-project data
directory, the checkpoint database, the recent-project list, the notes file
and the "which thread was I on" pointer.

No UI framework is imported here. A frontend calls these functions; it never
reaches for the filesystem itself.
"""

import json
import os
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

DB_FILENAME = "agent_workspace_memory.db"
CURRENT_THREAD_FILENAME = ".current_thread.txt"
NOTES_FILENAME = "notes.md"


def _data_root() -> Path:
    """Root of all agent data. Defaults to ./data when AGENT_DATA_ROOT is unset."""
    return Path(os.getenv("AGENT_DATA_ROOT") or "data")


def get_central_workspace_path(cwd: str | None = None) -> Path:
    """Data directory for the given working directory (defaults to os.getcwd())."""
    root = _data_root() / "central_workspace_data"
    dir_name = os.path.basename(cwd or os.getcwd())
    path = root / dir_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_db_path(cwd: str | None = None) -> Path:
    return get_central_workspace_path(cwd) / DB_FILENAME


# --- CHECKPOINTER -----------------------------------------------------------

# One SQLite connection per database file. Streamlit reruns the whole script on
# every interaction, so without this cache each rerun would open a new handle.
_savers: dict[str, SqliteSaver] = {}


def get_checkpointer(cwd: str | None = None) -> SqliteSaver:
    db_path = get_db_path(cwd)
    key = str(db_path)
    if key not in _savers:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        _savers[key] = SqliteSaver(conn)
    return _savers[key]


def _connect(cwd: str | None = None) -> sqlite3.Connection:
    return sqlite3.connect(get_db_path(cwd), check_same_thread=False)


# --- THREADS ----------------------------------------------------------------


def get_all_thread_ids(cwd: str | None = None) -> list[str]:
    """Every thread ID present in the checkpoint database."""
    if not get_db_path(cwd).exists():
        return []
    try:
        conn = _connect(cwd)
        try:
            rows = conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
        finally:
            conn.close()
        return [row[0] for row in rows]
    except Exception:
        return []


def delete_thread(thread_id: str, cwd: str | None = None) -> None:
    conn = _connect(cwd)
    try:
        conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        conn.commit()
    finally:
        conn.close()


def rename_thread(old_id: str, new_id: str, cwd: str | None = None) -> None:
    conn = _connect(cwd)
    try:
        conn.execute(
            "UPDATE checkpoints SET thread_id = ? WHERE thread_id = ?", (new_id, old_id)
        )
        conn.execute(
            "UPDATE writes SET thread_id = ? WHERE thread_id = ?", (new_id, old_id)
        )
        conn.commit()
    finally:
        conn.close()


def get_saved_thread_id(cwd: str | None = None) -> str | None:
    state_file = get_central_workspace_path(cwd) / CURRENT_THREAD_FILENAME
    if state_file.exists():
        try:
            return state_file.read_text().strip() or None
        except Exception:
            return None
    return None


def save_thread_id(thread_id: str, cwd: str | None = None) -> None:
    state_file = get_central_workspace_path(cwd) / CURRENT_THREAD_FILENAME
    try:
        state_file.write_text(thread_id)
    except Exception:
        pass


# --- RECENT PROJECTS --------------------------------------------------------


def history_file() -> Path:
    return _data_root() / "history_file" / ".gemini_agent_projects.json"


def load_recent_projects() -> list[str]:
    path = history_file()
    if path.exists():
        try:
            return json.load(open(path))
        except Exception:
            pass
    return []


def save_recent_project(path: str) -> None:
    projects = load_recent_projects()
    if path in projects:
        projects.remove(path)
    projects.insert(0, path)
    try:
        target = history_file()
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as f:
            json.dump(projects, f)
    except Exception:
        pass


# --- NOTES ------------------------------------------------------------------


def read_notes(cwd: str | None = None) -> str:
    notes = get_central_workspace_path(cwd) / NOTES_FILENAME
    return notes.read_text() if notes.exists() else ""


def write_notes(content: str, cwd: str | None = None) -> None:
    (get_central_workspace_path(cwd) / NOTES_FILENAME).write_text(content)


# --- FILES ------------------------------------------------------------------

SKIP_DIRS = {
    ".venv",
    ".git",
    "__pycache__",
    "node_modules",
    "workspace_data",
    "papers",
}


def list_project_files(root: str = ".") -> list[str]:
    """Project files, minus the noise directories no one wants to edit."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            found.append(os.path.join(dirpath, name))
    return found


def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


def write_file(path: str, content: str) -> str:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"
