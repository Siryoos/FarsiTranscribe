"""
Legacy terminal_display shim for backward compatibility.

Provides a minimal `terminal_display` object that forwards to the unified
terminal display implementation in `unified_terminal_display.py`.
"""

from typing import Optional

try:
    from .unified_terminal_display import create_preview_display
except Exception:
    create_preview_display = None  # type: ignore


class _TerminalDisplayShim:
    def __init__(self):
        self._display = None

    def _ensure_display(self):
        if self._display is None and create_preview_display is not None:
            # Create a simple display with 1 chunk and trivial duration
            self._display = create_preview_display(
                total_chunks=1, estimated_duration=1.0
            )

    def print_persian_preview(
        self, text: str, part_number: Optional[int] = None
    ):
        self._ensure_display()
        if self._display is not None and hasattr(
            self._display, "print_persian_preview"
        ):
            return self._display.print_persian_preview(text, part_number or 1)
        # Graceful fallback
        print(f"Part {part_number or 1} Preview: {text}")

    def print_simple_preview(self, text: str, label: str = ""):
        self._ensure_display()
        if self._display is not None and hasattr(
            self._display, "print_simple_preview"
        ):
            return self._display.print_simple_preview(text, label)
        # Graceful fallback
        prefix = f"{label}: " if label else ""
        print(f"{prefix}{text}")

    def check_unicode_support(self) -> bool:
        # Assume modern terminals support Unicode; unified display does internal checks
        return True


terminal_display = _TerminalDisplayShim()

__all__ = ["terminal_display"]
