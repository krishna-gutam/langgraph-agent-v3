"""Tests for the terminal frontend. The backend is mocked out - these only
check that the CLI drives the session correctly."""

import pytest
from unittest.mock import MagicMock, patch

from frontends import cli
from backend import Overseer, PendingToolCall


def make_session(pending=None, chunks=("Hi there!",)):
    session = MagicMock()
    session.is_ready.return_value = True
    session.stream.return_value = list(chunks)
    # pending_tool_calls() is polled repeatedly: return the calls once, then [].
    session.pending_tool_calls.side_effect = [pending or [], []]
    return session


@patch("frontends.cli.input", side_effect=["exit"])
def test_run_cli_exit(mock_input):
    session = make_session()
    cli.run_cli(session)
    session.stream.assert_not_called()


@patch("frontends.cli.input", side_effect=["Hello", "exit"])
def test_run_cli_streams_reply(mock_input, capsys):
    session = make_session()
    cli.run_cli(session)
    session.stream.assert_called_once_with("Hello")
    assert "Hi there!" in capsys.readouterr().out


@patch("frontends.cli.input", side_effect=["Run tool", "y", "exit"])
def test_run_cli_tool_approval(mock_input):
    call = PendingToolCall(id="1", name="run_bash", args={"script": "ls"})
    session = make_session(pending=[call])
    cli.run_cli(session)
    session.approve_tools.assert_called_once()
    session.deny_tools.assert_not_called()


@patch("frontends.cli.input", side_effect=["Run tool", "n", "exit"])
def test_run_cli_tool_denial(mock_input):
    call = PendingToolCall(id="1", name="run_bash", args={"script": "ls"})
    session = make_session(pending=[call])
    cli.run_cli(session)
    session.deny_tools.assert_called_once()
    session.approve_tools.assert_not_called()


@patch("frontends.cli.input", side_effect=["Run tool", "f", "use ls -la instead", "exit"])
def test_run_cli_tool_feedback(mock_input):
    call = PendingToolCall(id="1", name="run_bash", args={"script": "ls"})
    session = make_session(pending=[call])
    cli.run_cli(session)
    session.send_tool_feedback.assert_called_once_with("use ls -la instead")
    session.resume.assert_called_once()


@patch("frontends.cli.input", side_effect=["Run tool", "exit"])
def test_run_cli_auto_approve(mock_input):
    call = PendingToolCall(id="1", name="run_bash", args={"script": "ls"})
    session = make_session(pending=[call])
    cli.run_cli(session, auto_approve=True)
    session.approve_tools.assert_called_once()


# --- commands ---------------------------------------------------------------


def test_command_clear():
    session, overseer = MagicMock(), Overseer()
    overseer.start("goal")
    assert cli.handle_command(session, overseer, "/clear") is True
    session.clear_history.assert_called_once()
    assert overseer.active is False


def test_command_undo():
    session, overseer = MagicMock(), Overseer()
    cli.handle_command(session, overseer, "/undo")
    session.undo_last_turn.assert_called_once()


def test_command_new_thread_with_id():
    session, overseer = MagicMock(), Overseer()
    cli.handle_command(session, overseer, "/new feature-x")
    session.new_thread.assert_called_once_with("feature-x")


def test_plain_text_is_not_a_command():
    assert cli.handle_command(MagicMock(), Overseer(), "hello") is False


# --- overseer loop ----------------------------------------------------------


def test_run_overseer_runs_requested_steps():
    session = make_session()
    session.pending_tool_calls.side_effect = None
    session.pending_tool_calls.return_value = []
    overseer = Overseer()
    with patch.object(Overseer, "advance", return_value="do the thing"):
        cli.run_overseer(session, overseer, "ship it", steps=3)
    assert session.stream.call_count == 3


def test_run_overseer_stops_on_failure():
    session = make_session()
    session.pending_tool_calls.return_value = []
    overseer = Overseer()
    with patch.object(Overseer, "advance", side_effect=RuntimeError("no key")):
        cli.run_overseer(session, overseer, "ship it", steps=3)
    session.stream.assert_not_called()
