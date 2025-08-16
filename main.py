#!/usr/bin/env python3
"""Entry point that forwards to the modern src CLI."""

from src.core.cli import main  # noqa: F401


if __name__ == "__main__":
    main()
