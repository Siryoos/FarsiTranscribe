"""
Unified Terminal Display for Persian Transcription
Consolidates functionality from multiple terminal display modules.
Provides comprehensive terminal output with RTL support, Persian text handling,
and enhanced real-time preview display for transcription progress.
"""

import os
import sys
import logging
import time
import threading
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime, timedelta
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

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


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


@dataclass
class ChunkInfo:
    """Information about a transcription chunk."""

    chunk_id: int
    text: str
    start_time: float
    end_time: float
    duration: float
    progress: float = 0.0
    status: str = "pending"  # pending, transcribing, completed, failed
    speaker_id: Optional[str] = None


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
        """Manage text cache size."""
        if len(self.text_cache) >= self.cache_size_limit:
            # Remove oldest entries
            oldest_keys = list(self.text_cache.keys())[:100]
            for old_key in oldest_keys:
                del self.text_cache[old_key]

        self.text_cache[key] = value


class TerminalCapabilityDetector:
    """Detect terminal capabilities for optimal display mode selection."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def detect_capabilities(self) -> TerminalCapabilities:
        """Detect terminal capabilities."""
        try:
            # Check terminal width
            try:
                width = os.get_terminal_size().columns
            except OSError:
                width = 80

            # Check Unicode support
            unicode_support = self._test_unicode_support()

            # Check RTL support
            rtl_support = self._test_rtl_support()

            # Check color support
            color_support = self._test_color_support()

            return TerminalCapabilities(
                unicode_support=unicode_support,
                rtl_support=rtl_support,
                color_support=color_support,
                width=width,
                rich_available=RICH_AVAILABLE,
                rtl_libs_available=RTL_LIBS_AVAILABLE,
            )

        except Exception as e:
            self.logger.warning(f"Capability detection failed: {e}")
            return TerminalCapabilities(
                unicode_support=False,
                rtl_support=False,
                color_support=False,
                width=80,
                rich_available=False,
                rtl_libs_available=False,
            )

    def _test_unicode_support(self) -> bool:
        """Test Unicode support."""
        try:
            # Test Persian text display
            test_text = "سلام"
            print(test_text, end="", flush=True)
            print("\r", end="", flush=True)  # Return to start of line
            return True
        except Exception:
            return False

    def _test_rtl_support(self) -> bool:
        """Test RTL text support."""
        try:
            if not RTL_LIBS_AVAILABLE:
                return False

            # Test RTL processing
            test_text = "سلام دنیا"
            processor = RTLTextProcessor()
            processed = processor.process_persian_text(test_text)
            return processed != test_text
        except Exception:
            return False

    def _test_color_support(self) -> bool:
        """Test color support."""
        try:
            # Check if we're in a terminal that supports colors
            return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        except Exception:
            return False


class UnifiedTerminalDisplay:
    """Unified terminal display with RTL support and enhanced preview capabilities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # Detect capabilities
        detector = TerminalCapabilityDetector()
        self.capabilities = detector.detect_capabilities()

        # Select display mode
        self.display_mode = self._select_display_mode()

        # Initialize components
        self.console = (
            Console()
            if RICH_AVAILABLE and self.capabilities.rich_available
            else None
        )
        self.rtl_processor = RTLTextProcessor()

        # Log capabilities
        self._log_capabilities()

    def _select_display_mode(self) -> DisplayMode:
        """Select optimal display mode based on capabilities."""
        if self.capabilities.rtl_support and self.capabilities.rich_available:
            return DisplayMode.RTL_RICH
        elif self.capabilities.rtl_support:
            return DisplayMode.RTL_PLAIN
        elif self.capabilities.rich_available:
            return DisplayMode.LTR_RICH
        elif self.capabilities.unicode_support:
            return DisplayMode.LTR_PLAIN
        else:
            return DisplayMode.FALLBACK

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
            print(f"\n{title}")
            print("=" * len(title))

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


class EnhancedPreviewDisplay:
    """Enhanced preview display with real-time progress tracking."""

    def __init__(self, total_chunks: int, estimated_duration: float = None):
        self.total_chunks = total_chunks
        self.estimated_duration = estimated_duration
        self.chunks: Dict[int, ChunkInfo] = {}
        self.current_chunk = 0
        self.start_time = time.time()
        self.console = Console() if RICH_AVAILABLE else None

        # Display settings
        self.max_preview_length = 80
        self.show_timing = True
        self.show_progress_bars = True
        self.show_speaker_icons = True

        # Threading
        self._lock = threading.Lock()
        self._display_thread = None
        self._stop_display = False

    def add_chunk(
        self,
        chunk_id: int,
        start_time: float,
        end_time: float,
        duration: float,
        speaker_id: Optional[str] = None,
    ):
        """Add a new chunk to track."""
        with self._lock:
            self.chunks[chunk_id] = ChunkInfo(
                chunk_id=chunk_id,
                text="",
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                speaker_id=speaker_id,
            )

    def update_chunk_progress(
        self, chunk_id: int, progress: float, text: str = ""
    ):
        """Update chunk transcription progress."""
        with self._lock:
            if chunk_id in self.chunks:
                self.chunks[chunk_id].progress = progress
                if text:
                    self.chunks[chunk_id].text = text
                self.chunks[chunk_id].status = (
                    "transcribing" if progress < 100 else "completed"
                )

    def set_current_chunk(self, chunk_id: int):
        """Set the currently active chunk."""
        with self._lock:
            self.current_chunk = chunk_id

    def format_time(self, seconds: float) -> str:
        """Format time in MM:SS format."""
        return str(timedelta(seconds=int(seconds)))

    def format_duration(self, duration: float) -> str:
        """Format duration with appropriate units."""
        if duration < 60:
            return f"{duration:.2f}s"
        elif duration < 3600:
            minutes = duration / 60
            return f"{minutes:.1f}m"
        else:
            hours = duration / 3600
            return f"{hours:.1f}h"

    def get_speaker_icon(self, speaker_id: Optional[str] = None) -> str:
        """Get speaker icon for display."""
        if not self.show_speaker_icons:
            return ""

        if speaker_id is None:
            return "🔊"
        else:
            return f"👤{speaker_id}"

    def create_chunk_display(self, chunk: ChunkInfo) -> str:
        """Create display string for a single chunk."""
        # Progress bar
        progress_bar = ""
        if self.show_progress_bars and chunk.status == "transcribing":
            progress = int(chunk.progress / 10)
            progress_bar = "█" * progress + "░" * (10 - progress)

        # Text preview
        text_preview = chunk.text[: self.max_preview_length]
        if len(chunk.text) > self.max_preview_length:
            text_preview += "..."

        # Speaker icon
        speaker_icon = self.get_speaker_icon(chunk.speaker_id)

        # Status indicator
        status_indicator = ""
        if chunk.status == "transcribing":
            status_indicator = "🔄"
        elif chunk.status == "completed":
            status_indicator = "✅"
        elif chunk.status == "failed":
            status_indicator = "❌"

        # Format the display
        return f"{status_indicator} Transcribing: {chunk.progress:.0f}% {progress_bar}\nChunk {chunk.chunk_id} {speaker_icon}\n{text_preview}"

    def create_progress_overview(self) -> str:
        """Create progress overview display."""
        completed = sum(
            1 for c in self.chunks.values() if c.status == "completed"
        )
        transcribing = sum(
            1 for c in self.chunks.values() if c.status == "transcribing"
        )
        pending = sum(1 for c in self.chunks.values() if c.status == "pending")

        elapsed_time = time.time() - self.start_time
        elapsed_str = self.format_time(elapsed_time)

        if self.estimated_duration:
            remaining = max(0, self.estimated_duration - elapsed_time)
            remaining_str = self.format_time(remaining)
            eta_str = f"[{elapsed_str}<{remaining_str})"
        else:
            eta_str = f"[{elapsed_str})"

        overview_lines = []
        for chunk_id in sorted(self.chunks.keys()):
            chunk = self.chunks[chunk_id]
            if chunk.status == "completed":
                overview_lines.append(
                    f"| {chunk_id}/{self.total_chunks} {eta_str} {self.format_duration(chunk.duration)}/chunk"
                )
            elif chunk.status == "transcribing":
                overview_lines.append(
                    f"| {chunk_id}/{self.total_chunks} {eta_str} {self.format_duration(chunk.duration)}/chunk 🔄"
                )
            else:
                overview_lines.append(
                    f"| {chunk_id}/{self.total_chunks} {eta_str} {self.format_duration(chunk.duration)}/chunk ⏳"
                )

        return "\n".join(overview_lines)

    def display(self):
        """Display the current preview state."""
        if not self.chunks:
            return

        # Clear screen for better display
        os.system("clear" if os.name == "posix" else "cls")

        # Header
        print("🎙️  FarsiTranscribe - Real-time Preview")
        print("=" * 60)

        # Left section - Transcription details
        print("📝 Transcription Details:")
        print("-" * 40)

        # Show recent chunks (last 20)
        recent_chunks = sorted(self.chunks.keys(), reverse=True)[:20]
        for chunk_id in recent_chunks:
            chunk = self.chunks[chunk_id]
            if chunk.text.strip():  # Only show chunks with text
                print(self.create_chunk_display(chunk))
                print()

        # Right section - Progress overview
        print("\n📊 Progress Overview:")
        print("-" * 40)
        print(self.create_progress_overview())

        # Footer
        print("\n" + "=" * 60)
        print("💡 Ctrl+K to generate a command")

        # Current status
        elapsed = time.time() - self.start_time
        completed = sum(
            1 for c in self.chunks.values() if c.status == "completed"
        )
        progress_percent = (
            (completed / self.total_chunks) * 100
            if self.total_chunks > 0
            else 0
        )

        print(
            f"⏱️  Elapsed: {self.format_time(elapsed)} | Progress: {progress_percent:.1f}% ({completed}/{self.total_chunks})"
        )

    def start_display_thread(self):
        """Start the display update thread."""
        if self._display_thread and self._display_thread.is_alive():
            return

        self._stop_display = False
        self._display_thread = threading.Thread(
            target=self._display_loop, daemon=True
        )
        self._display_thread.start()

    def stop_display_thread(self):
        """Stop the display update thread."""
        self._stop_display = True
        if self._display_thread and self._display_thread.is_alive():
            self._display_thread.join(timeout=1.0)

    def _display_loop(self):
        """Main display update loop."""
        while not self._stop_display:
            try:
                self.display()
                time.sleep(0.5)  # Update every 500ms
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Display error: {e}")
                break

    def __enter__(self):
        """Context manager entry."""
        self.start_display_thread()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_display_thread()


class SimplePreviewDisplay:
    """Simple fallback preview display for terminals without Rich."""

    def __init__(self, total_chunks: int):
        self.total_chunks = total_chunks
        self.chunks = {}
        self.current_chunk = 0
        self.start_time = time.time()

    def add_chunk(
        self,
        chunk_id: int,
        start_time: float,
        end_time: float,
        duration: float,
        speaker_id: Optional[str] = None,
    ):
        """Add a new chunk."""
        self.chunks[chunk_id] = {
            "id": chunk_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "text": "",
            "progress": 0,
            "speaker_id": speaker_id,
        }

    def update_chunk_progress(
        self, chunk_id: int, progress: float, text: str = ""
    ):
        """Update chunk progress."""
        if chunk_id in self.chunks:
            self.chunks[chunk_id]["progress"] = progress
            if text:
                self.chunks[chunk_id]["text"] = text

    def display(self):
        """Display current state."""
        os.system("clear" if os.name == "posix" else "cls")

        print("🎙️  FarsiTranscribe - Simple Preview")
        print("=" * 50)

        for chunk_id in sorted(self.chunks.keys()):
            chunk = self.chunks[chunk_id]
            progress_bar = "█" * int(chunk["progress"] / 10) + "░" * (
                10 - int(chunk["progress"] / 10)
            )
            speaker_icon = (
                f"👤{chunk['speaker_id']}" if chunk["speaker_id"] else "🔊"
            )

            print(f"🔄 Transcribing: {chunk['progress']:.0f}% {progress_bar}")
            print(f"Chunk {chunk_id} {speaker_icon}")
            if chunk["text"]:
                preview = (
                    chunk["text"][:80] + "..."
                    if len(chunk["text"]) > 80
                    else chunk["text"]
                )
                print(f"{preview}")
            print()

        print("=" * 50)
        print("💡 Ctrl+K to generate a command")


def create_preview_display(
    total_chunks: int,
    estimated_duration: float = None,
    use_enhanced: bool = True,
):
    """Factory function to create appropriate preview display."""
    if use_enhanced and RICH_AVAILABLE:
        return EnhancedPreviewDisplay(total_chunks, estimated_duration)
    else:
        return SimplePreviewDisplay(total_chunks)


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
