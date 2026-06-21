"""Command-line interface for the Label Studio workflow scripts."""

from __future__ import annotations

import sys

from .upload_schema import main as upload_schema_main
from .upload_tasks import main as upload_tasks_main


def main(argv: list[str] | None = None) -> int:
    """Run a Label Studio workflow command."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_usage()
        return 0

    command, *command_args = args
    if command == "upload-schema":
        if command_args:
            _print_usage()
            return 1
        return upload_schema_main()
    if command == "upload-tasks":
        return upload_tasks_main(command_args)

    print(f"Unknown command: {command}")  # noqa: T201
    _print_usage()
    return 1


def _print_usage() -> None:
    print(  # noqa: T201
        "Usage: uv run --group label-studio python -m scripts.label_studio.cli "
        "<upload-schema|upload-tasks> [paths...]"
    )


if __name__ == "__main__":
    raise SystemExit(main())
