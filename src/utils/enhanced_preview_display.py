#!/usr/bin/env python3
"""
Enhanced Preview Display System for FarsiTranscribe.
Provides real-time transcription progress with detailed chunk information.
"""

import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import sys
import os

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.layout import Layout
    from rich.live import Live
    from rich.columns import Columns
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


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
        
    def add_chunk(self, chunk_id: int, start_time: float, end_time: float, 
                  duration: float, speaker_id: Optional[str] = None):
        """Add a new chunk to track."""
        with self._lock:
            self.chunks[chunk_id] = ChunkInfo(
                chunk_id=chunk_id,
                text="",
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                speaker_id=speaker_id
            )
    
    def update_chunk_progress(self, chunk_id: int, progress: float, text: str = ""):
        """Update chunk transcription progress."""
        with self._lock:
            if chunk_id in self.chunks:
                self.chunks[chunk_id].progress = progress
                if text:
                    self.chunks[chunk_id].text = text
                self.chunks[chunk_id].status = "transcribing" if progress < 100 else "completed"
    
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
        text_preview = chunk.text[:self.max_preview_length]
        if len(chunk.text) > self.max_preview_length:
            text_preview += "..."
        
        # Speaker icon
        speaker_icon = self.get_speaker_icon(chunk.speaker_id)
        
        # Timing info
        timing_info = ""
        if self.show_timing:
            start_str = self.format_time(chunk.start_time)
            end_str = self.format_time(chunk.end_time)
            duration_str = self.format_duration(chunk.duration)
            timing_info = f"[{start_str}<{end_str}) {duration_str}/chunk"
        
        # Status indicator
        status_indicator = ""
        if chunk.status == "transcribing":
            status_indicator = "🔄"
        elif chunk.status == "completed":
            status_indicator = "✅"
        elif chunk.status == "failed":
            status_indicator = "❌"
        
        # Format the display
        if RICH_AVAILABLE:
            return f"{status_indicator} Transcribing: {chunk.progress:.0f}% {progress_bar}\nChunk {chunk.chunk_id} {speaker_icon}\n{text_preview}"
        else:
            return f"{status_indicator} Transcribing: {chunk.progress:.0f}% {progress_bar}\nChunk {chunk.chunk_id} {speaker_icon}\n{text_preview}"
    
    def create_progress_overview(self) -> str:
        """Create progress overview display."""
        completed = sum(1 for c in self.chunks.values() if c.status == "completed")
        transcribing = sum(1 for c in self.chunks.values() if c.status == "transcribing")
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
                overview_lines.append(f"| {chunk_id}/{self.total_chunks} {eta_str} {self.format_duration(chunk.duration)}/chunk")
            elif chunk.status == "transcribing":
                overview_lines.append(f"| {chunk_id}/{self.total_chunks} {eta_str} {self.format_duration(chunk.duration)}/chunk 🔄")
            else:
                overview_lines.append(f"| {chunk_id}/{self.total_chunks} {eta_str} {self.format_duration(chunk.duration)}/chunk ⏳")
        
        return "\n".join(overview_lines)
    
    def display(self):
        """Display the current preview state."""
        if not self.chunks:
            return
        
        # Clear screen for better display
        os.system('clear' if os.name == 'posix' else 'cls')
        
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
        completed = sum(1 for c in self.chunks.values() if c.status == "completed")
        progress_percent = (completed / self.total_chunks) * 100 if self.total_chunks > 0 else 0
        
        print(f"⏱️  Elapsed: {self.format_time(elapsed)} | Progress: {progress_percent:.1f}% ({completed}/{self.total_chunks})")
    
    def start_display_thread(self):
        """Start the display update thread."""
        if self._display_thread and self._display_thread.is_alive():
            return
        
        self._stop_display = False
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
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
    
    def add_chunk(self, chunk_id: int, start_time: float, end_time: float, 
                  duration: float, speaker_id: Optional[str] = None):
        """Add a new chunk."""
        self.chunks[chunk_id] = {
            'id': chunk_id,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'text': '',
            'progress': 0,
            'speaker_id': speaker_id
        }
    
    def update_chunk_progress(self, chunk_id: int, progress: float, text: str = ""):
        """Update chunk progress."""
        if chunk_id in self.chunks:
            self.chunks[chunk_id]['progress'] = progress
            if text:
                self.chunks[chunk_id]['text'] = text
    
    def display(self):
        """Display current state."""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🎙️  FarsiTranscribe - Simple Preview")
        print("=" * 50)
        
        for chunk_id in sorted(self.chunks.keys()):
            chunk = self.chunks[chunk_id]
            progress_bar = "█" * int(chunk['progress'] / 10) + "░" * (10 - int(chunk['progress'] / 10))
            speaker_icon = f"👤{chunk['speaker_id']}" if chunk['speaker_id'] else "🔊"
            
            print(f"🔄 Transcribing: {chunk['progress']:.0f}% {progress_bar}")
            print(f"Chunk {chunk_id} {speaker_icon}")
            if chunk['text']:
                preview = chunk['text'][:80] + "..." if len(chunk['text']) > 80 else chunk['text']
                print(f"{preview}")
            print()
        
        print("=" * 50)
        print("💡 Ctrl+K to generate a command")


def create_preview_display(total_chunks: int, estimated_duration: float = None, 
                         use_enhanced: bool = True):
    """Factory function to create appropriate preview display."""
    if use_enhanced and RICH_AVAILABLE:
        return EnhancedPreviewDisplay(total_chunks, estimated_duration)
    else:
        return SimplePreviewDisplay(total_chunks)
