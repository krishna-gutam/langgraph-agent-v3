"""Launcher for the terminal frontend: `python cli.py [--help]`."""

from frontends.cli import main, run_cli  # noqa: F401  (re-exported for convenience)

if __name__ == "__main__":
    main()
