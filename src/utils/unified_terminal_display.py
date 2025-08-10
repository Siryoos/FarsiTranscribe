"""
Unified Terminal Display for Persian Transcription
Consolidates functionality from multiple terminal display modules.
Provides comprehensive terminal output with RTL support and Persian text handling.
"""

import os
import sys
import logging
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from rich.table import Table

# Optional imports with fallbacks
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
    )
    from rich.panel import Panel
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import arabic_reshaper
    from bidi.algorithm import get_display

    RTL_LIBS_AVAILABLE = True
except ImportError:
    RTL_LIBS_AVAILABLE = False


class DisplayMode(Enum):
    """Available display modes for terminal output."""

    RTL_RICH = "rtl_rich"  # Full RTL support with Rich formatting
    RTL_PLAIN = "rtl_plain"  # RTL support with plain text
    LTR_RICH = "ltr_rich"  # Left-to-right with Rich formatting
    LTR_PLAIN = "ltr_plain"  # Basic left-to-right plain text
    FALLBACK = "fallback"  # Minimal fallback mode


@dataclass
class TerminalCapabilities:
    """Terminal capability detection results."""

    unicode_support: bool
    rtl_support: bool
    color_support: bool
    width: int
    rich_available: bool
    rtl_libs_available: bool


class RTLTextProcessor:
    """RTL text processing utilities."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.text_cache = {}
        self.cache_size_limit = 1000

    def process_persian_text(self, text: str) -> str:
        """Process Persian text for RTL display."""
        if not text:
            return text

        # Check cache first
        if text in self.text_cache:
            return self.text_cache[text]

        try:
            if RTL_LIBS_AVAILABLE:
                # Reshape Arabic/Persian text
                reshaped_text = arabic_reshaper.reshape(text)
                # Apply bidirectional algorithm
                processed_text = get_display(reshaped_text)
            else:
                # Fallback: basic RTL handling
                processed_text = text[::-1]  # Simple reverse

            # Cache the result
            self._manage_cache(text, processed_text)
            return processed_text

        except Exception as e:
            self.logger.warning(f"RTL text processing failed: {e}")
            return text

    def _manage_cache(self, key: str, value: str):
        """Manage text processing cache."""
        if len(self.text_cache) >= self.cache_size_limit:
            # Remove oldest entry
            oldest_key = next(iter(self.text_cache))
            del self.text_cache[oldest_key]

        self.text_cache[key] = value


class TerminalCapabilityDetector:
    """Detect terminal capabilities."""

    @staticmethod
    def detect_capabilities() -> TerminalCapabilities:
        """Detect terminal capabilities."""
        return TerminalCapabilities(
            unicode_support=TerminalCapabilityDetector._test_unicode_support(),
            rtl_support=TerminalCapabilityDetector._test_rtl_support(),
            color_support=TerminalCapabilityDetector._test_color_support(),
            width=TerminalCapabilityDetector._get_terminal_width(),
            rich_available=RICH_AVAILABLE,
            rtl_libs_available=RTL_LIBS_AVAILABLE,
        )

    @staticmethod
    def _test_unicode_support() -> bool:
        """Test Unicode support."""
        try:
            # Test Persian text output
            test_text = "سلام دنیا"
            print(test_text, end="", flush=True)
            print("\r", end="", flush=True)  # Clear line
            return True
        except UnicodeEncodeError:
            return False

    @staticmethod
    def _test_rtl_support() -> bool:
        """Test RTL support."""
        try:
            # Test RTL text
            test_text = "سلام دنیا"
            if RTL_LIBS_AVAILABLE:
                reshaped = arabic_reshaper.reshape(test_text)
                rtl_text = get_display(reshaped)
                print(rtl_text, end="", flush=True)
                print("\r", end="", flush=True)
                return True
            return False
        except Exception:
            return False

    @staticmethod
    def _test_color_support() -> bool:
        """Test color support."""
        try:
            # Test ANSI color codes
            print("\033[32mTest\033[0m", end="", flush=True)
            print("\r", end="", flush=True)
            return True
        except Exception:
            return False

    @staticmethod
    def _get_terminal_width() -> int:
        """Get terminal width."""
        try:
            return os.get_terminal_size().columns
        except Exception:
            return 80  # Default fallback


class UnifiedTerminalDisplay:
    """
    Unified terminal display with comprehensive Persian text support.
    Consolidates functionality from multiple terminal display modules.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.capabilities = TerminalCapabilityDetector.detect_capabilities()
        self.display_mode = self._determine_optimal_display_mode()
        self.rtl_processor = RTLTextProcessor()

        # Initialize Rich console if available
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None

        self._setup_unicode_environment()
        self._log_capabilities()

    def _determine_optimal_display_mode(self) -> DisplayMode:
        """Determine the optimal display mode based on capabilities."""
        if (
            self.capabilities.rich_available
            and self.capabilities.rtl_libs_available
            and self.capabilities.rtl_support
        ):
            return DisplayMode.RTL_RICH
        elif (
            self.capabilities.rtl_libs_available
            and self.capabilities.rtl_support
        ):
            return DisplayMode.RTL_PLAIN
        elif self.capabilities.rich_available:
            return DisplayMode.LTR_RICH
        elif self.capabilities.unicode_support:
            return DisplayMode.LTR_PLAIN
        else:
            return DisplayMode.FALLBACK

    def _setup_unicode_environment(self):
        """Setup Unicode environment for proper text display."""
        try:
            # Set UTF-8 encoding
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8")

            # Set environment variables
            os.environ["PYTHONIOENCODING"] = "utf-8"
            os.environ["LC_ALL"] = "en_US.UTF-8"

        except Exception as e:
            self.logger.warning(f"Unicode environment setup failed: {e}")

    def _log_capabilities(self):
        """Log terminal capabilities."""
        self.logger.info(
            f"Terminal capabilities: Unicode={self.capabilities.unicode_support}, "
            f"RTL={self.capabilities.rtl_support}, Rich={self.capabilities.rich_available}, "
            f"Width={self.capabilities.width}"
        )
        self.logger.info(f"Display mode: {self.display_mode.value}")

    def print_persian_preview(
        self, text: str, part_number: int, max_length: int = 100
    ):
        """Print Persian text preview with appropriate formatting."""
        if not text:
            return

        # Process text for display
        display_text = self._process_text_for_display(text, max_length)

        # Print based on display mode
        if self.display_mode == DisplayMode.RTL_RICH:
            self._print_rtl_rich_preview(display_text, part_number)
        elif self.display_mode == DisplayMode.RTL_PLAIN:
            self._print_rtl_plain_preview(display_text, part_number)
        elif self.display_mode == DisplayMode.LTR_RICH:
            self._print_ltr_rich_preview(display_text, part_number)
        elif self.display_mode == DisplayMode.LTR_PLAIN:
            self._print_ltr_plain_preview(display_text, part_number)
        else:
            self._print_fallback_preview(display_text, part_number)

    def _process_text_for_display(self, text: str, max_length: int) -> str:
        """Process text for display with length limits."""
        if len(text) <= max_length:
            return text

        # Truncate and add ellipsis
        truncated = text[: max_length - 3] + "..."
        return truncated

    def _print_rtl_rich_preview(self, text: str, part_number: int):
        """Print RTL preview with Rich formatting."""
        if not self.console:
            return

        # Process text for RTL
        rtl_text = self.rtl_processor.process_persian_text(text)

        # Create rich text
        rich_text = Text(rtl_text, style="cyan")

        # Create panel
        panel = Panel(
            rich_text,
            title=f"بخش {part_number}",
            border_style="blue",
            padding=(0, 1),
        )

        self.console.print(panel)

    def _print_rtl_plain_preview(self, text: str, part_number: int):
        """Print RTL preview with plain text."""
        # Process text for RTL
        rtl_text = self.rtl_processor.process_persian_text(text)

        # Print with simple formatting
        print(f"بخش {part_number}: {rtl_text}")

    def _print_ltr_rich_preview(self, text: str, part_number: int):
        """Print LTR preview with Rich formatting."""
        if not self.console:
            return

        # Create rich text
        rich_text = Text(text, style="cyan")

        # Create panel
        panel = Panel(
            rich_text,
            title=f"Part {part_number}",
            border_style="blue",
            padding=(0, 1),
        )

        self.console.print(panel)

    def _print_ltr_plain_preview(self, text: str, part_number: int):
        """Print LTR preview with plain text."""
        print(f"Part {part_number}: {text}")

    def _print_fallback_preview(self, text: str, part_number: int):
        """Print fallback preview."""
        print(f"Part {part_number}: {text}")

    def print_simple_preview(self, text: str, part_number: str):
        """Print simple preview without complex formatting."""
        if not text:
            return

        # Truncate text for preview
        preview_text = text[:80] + "..." if len(text) > 80 else text

        if self.display_mode in [DisplayMode.RTL_RICH, DisplayMode.RTL_PLAIN]:
            # RTL mode
            rtl_text = self.rtl_processor.process_persian_text(preview_text)
            print(f"بخش {part_number}: {rtl_text}")
        else:
            # LTR mode
            print(f"Part {part_number}: {preview_text}")

    def print_header(self, title: str):
        """Print header with title."""
        if (
            self.display_mode in [DisplayMode.RTL_RICH, DisplayMode.LTR_RICH]
            and self.console
        ):
            # Rich formatting
            header_text = Text(title, style="bold magenta")
            panel = Panel(header_text, border_style="green")
            self.console.print(panel)
        else:
            # Plain text
            print("=" * 60)
            print(title)
            print("=" * 60)

    def print_success(self, message: str):
        """Print success message."""
        if self.console:
            self.console.print(f"✅ {message}", style="bold green")
        else:
            print(f"✅ {message}")

    def print_error(self, message: str):
        """Print error message."""
        if self.console:
            self.console.print(f"❌ {message}", style="bold red")
        else:
            print(f"❌ {message}")

    def print_warning(self, message: str):
        """Print warning message."""
        if self.console:
            self.console.print(f"⚠️  {message}", style="bold yellow")
        else:
            print(f"⚠️  {message}")

    def print_info(self, message: str):
        """Print info message."""
        if self.console:
            self.console.print(f"ℹ️  {message}", style="blue")
        else:
            print(f"ℹ️  {message}")

    def create_progress_bar(self, total: int, description: str = "Processing"):
        """Create progress bar."""
        if RICH_AVAILABLE and self.console:
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console,
            )
        else:
            # Fallback progress bar
            class DummyProgress:
                def __init__(self):
                    self.task_id = None

                def add_task(self, description, total=None):
                    self.task_id = 1
                    print(f"{description}: 0/{total}")
                    return self.task_id

                def update(self, task_id, advance=1):
                    print(f"Progress: {advance}")

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    print("Completed")

            return DummyProgress()

    def format_transcription_table(
        self, transcriptions: List[str]
    ) -> Optional["Table"]:
        """Format transcriptions as a table."""
        if not RICH_AVAILABLE or not self.console:
            return None

        table = Table(title="Transcription Results")
        table.add_column("Part", style="cyan", no_wrap=True)
        table.add_column("Text", style="green")

        for i, transcription in enumerate(transcriptions, 1):
            # Process text for display
            display_text = self._process_text_for_display(transcription, 50)
            if self.display_mode in [
                DisplayMode.RTL_RICH,
                DisplayMode.RTL_PLAIN,
            ]:
                display_text = self.rtl_processor.process_persian_text(
                    display_text
                )

            table.add_row(f"Part {i}", display_text)

        return table

    def check_unicode_support(self) -> bool:
        """Check if terminal supports Unicode."""
        return self.capabilities.unicode_support

    def print_configuration_info(self):
        """Print configuration information."""
        self.print_header("Terminal Configuration")

        info_items = [
            f"Unicode Support: {'✅' if self.capabilities.unicode_support else '❌'}",
            f"RTL Support: {'✅' if self.capabilities.rtl_support else '❌'}",
            f"Color Support: {'✅' if self.capabilities.color_support else '❌'}",
            f"Rich Library: {'✅' if self.capabilities.rich_available else '❌'}",
            f"RTL Libraries: {'✅' if self.capabilities.rtl_libs_available else '❌'}",
            f"Terminal Width: {self.capabilities.width}",
            f"Display Mode: {self.display_mode.value}",
        ]

        for item in info_items:
            self.print_info(item)


def get_terminal_capabilities() -> Dict[str, bool]:
    """Get terminal capabilities summary."""
    detector = TerminalCapabilityDetector()
    caps = detector.detect_capabilities()

    return {
        "unicode_support": caps.unicode_support,
        "rtl_support": caps.rtl_support,
        "color_support": caps.color_support,
        "rich_available": caps.rich_available,
        "rtl_libs_available": caps.rtl_libs_available,
        "width": caps.width,
    }


def create_unified_display() -> UnifiedTerminalDisplay:
    """Create unified terminal display."""
    return UnifiedTerminalDisplay()
