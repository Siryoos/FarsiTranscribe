#!/usr/bin/env python3
"""Compatibility shim.

This module forwards to the official CLI entrypoint in
`farsi_transcribe.cli:main` to avoid duplication.
"""

from farsi_transcribe.cli import main  # noqa: F401


if __name__ == "__main__":
    main()
