"""
Enhanced terminal display utilities with Unicode support for Persian text.
"""

import os
import sys
from typing import List, Optional
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from colorama import init, Fore, Back, Style

# Initialize colorama for cross-platform color support
init(autoreset=True)


class TerminalDisplay:
    """Enhanced terminal display with Unicode support for Persian text."""
    
    def __init__(self):
        self.console = Console()
        self._setup_unicode_support()
    
    def _setup_unicode_support(self):
        """Setup Unicode support for Persian text display."""
        # Set environment variables for better Unicode support
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        # Try to set terminal encoding
        if hasattr(sys.stdout, 'reconfigure'):
            try:
                sys.stdout.reconfigure(encoding='utf-8')
                sys.stderr.reconfigure(encoding='utf-8')
            except:
                pass
    
    def print_persian_preview(self, text: str, part_number: int, max_length: int = 80):
        """Print Persian text preview with proper formatting."""
        if not text or not text.strip():
            return
        
        # Clean and format the text
        cleaned_text = text.strip()
        
        # Truncate if too long
        if len(cleaned_text) > max_length:
            cleaned_text = cleaned_text[:max_length] + "..."
        
        # Create rich text with Persian support
        rich_text = Text()
        rich_text.append(f"Part {part_number} Preview: ", style="bold cyan")
        rich_text.append(cleaned_text, style="white")
        
        # Create a panel for better visibility
        panel = Panel(
            rich_text,
            title="🎙️ Transcription Preview",
            border_style="blue",
            padding=(0, 1)
        )
        
        self.console.print(panel)
    
    def print_simple_preview(self, text: str, part_number: int):
        """Simple preview without rich formatting (fallback)."""
        if not text or not text.strip():
            return
        
        cleaned_text = text.strip()
        if len(cleaned_text) > 100:
            cleaned_text = cleaned_text[:100] + "..."
        
        print(f"{Fore.CYAN}Part {part_number} Preview:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{cleaned_text}{Style.RESET_ALL}")
        print("-" * 50)
    
    def create_progress_bar(self, total: int, description: str = "Processing"):
        """Create a rich progress bar."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        )
    
    def print_header(self, title: str):
        """Print a formatted header."""
        header_text = Text(title, style="bold white on blue")
        panel = Panel(header_text, border_style="blue")
        self.console.print(panel)
    
    def print_success(self, message: str):
        """Print a success message."""
        self.console.print(f"✅ {message}", style="bold green")
    
    def print_error(self, message: str):
        """Print an error message."""
        self.console.print(f"❌ {message}", style="bold red")
    
    def print_warning(self, message: str):
        """Print a warning message."""
        self.console.print(f"⚠️ {message}", style="bold yellow")
    
    def print_info(self, message: str):
        """Print an info message."""
        self.console.print(f"ℹ️ {message}", style="cyan")
    
    def format_transcription_table(self, transcriptions: List[str]) -> Table:
        """Create a table for displaying transcriptions."""
        table = Table(title="Transcription Results")
        table.add_column("Part", style="cyan", no_wrap=True)
        table.add_column("Text", style="white")
        table.add_column("Length", style="green")
        
        for i, transcription in enumerate(transcriptions, 1):
            if transcription.strip():
                preview = transcription[:50] + "..." if len(transcription) > 50 else transcription
                table.add_row(str(i), preview, str(len(transcription)))
        
        return table
    
    def check_unicode_support(self) -> bool:
        """Check if terminal supports Unicode properly."""
        try:
            # Test Persian characters
            test_text = "سلام دنیا"
            self.console.print(test_text)
            return True
        except UnicodeEncodeError:
            return False


# Global instance for easy access
terminal_display = TerminalDisplay() 