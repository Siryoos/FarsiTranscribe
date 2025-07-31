"""
Enhanced terminal display utilities with RTL support for Persian text.
Backward compatible with enhanced RTL display system.
"""

import os
import sys
from typing import List, Optional

# Import the enhanced RTL display system
try:
    from .rtl_terminal_display import enhanced_rtl_display
    RTL_ENHANCED_AVAILABLE = True
except ImportError:
    RTL_ENHANCED_AVAILABLE = False

# Fallback imports for backward compatibility
try:
    from rich.console import Console
    from rich.text import Text
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False


class TerminalDisplay:
    """Enhanced terminal display with RTL support for Persian text."""
    
    def __init__(self):
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
        self._setup_unicode_support()
        
        # Print configuration info if enhanced RTL is available
        if RTL_ENHANCED_AVAILABLE:
            print("\n🔧 Enhanced RTL Display System Activated")
    
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
        """Print Persian text preview with enhanced RTL support."""
        if RTL_ENHANCED_AVAILABLE:
            # Use the enhanced RTL display system
            enhanced_rtl_display.print_persian_preview(text, part_number, max_length)
            return
        
        # Fallback to original implementation
        if not text or not text.strip():
            return
        
        cleaned_text = text.strip()
        if len(cleaned_text) > max_length:
            cleaned_text = cleaned_text[:max_length] + "..."
        
        if RICH_AVAILABLE and self.console:
            rich_text = Text()
            rich_text.append(f"Part {part_number} Preview: ", style="bold cyan")
            rich_text.append(cleaned_text, style="white")
            
            panel = Panel(
                rich_text,
                title="🎙️ Transcription Preview",
                border_style="blue",
                padding=(0, 1)
            )
            
            self.console.print(panel)
        else:
            # Plain text fallback
            print(f"Part {part_number} Preview: {cleaned_text}")
    
    def print_simple_preview(self, text: str, part_number: int):
        """Simple preview with enhanced RTL support."""
        if RTL_ENHANCED_AVAILABLE:
            # Use the enhanced RTL display system
            enhanced_rtl_display.print_simple_preview(text, str(part_number))
            return
        
        # Fallback implementation
        if not text or not text.strip():
            return
        
        cleaned_text = text.strip()
        if len(cleaned_text) > 100:
            cleaned_text = cleaned_text[:100] + "..."
        
        if COLORAMA_AVAILABLE:
            print(f"{Fore.CYAN}Part {part_number} Preview:{Style.RESET_ALL}")
            print(f"{Fore.WHITE}{cleaned_text}{Style.RESET_ALL}")
            print("-" * 50)
        else:
            print(f"Part {part_number} Preview: {cleaned_text}")
            print("-" * 50)
    
    def create_progress_bar(self, total: int, description: str = "Processing"):
        """Create a rich progress bar."""
        if RICH_AVAILABLE and self.console:
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=self.console
            )
        else:
            # Return a dummy progress bar for compatibility
            class DummyProgress:
                def add_task(self, description, total=None):
                    return 0
                def update(self, task_id, advance=1):
                    pass
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return DummyProgress()
    
    def print_header(self, title: str):
        """Print a formatted header."""
        if RICH_AVAILABLE and self.console:
            header_text = Text(title, style="bold white on blue")
            panel = Panel(header_text, border_style="blue")
            self.console.print(panel)
        else:
            print("=" * 60)
            print(f"  {title}")
            print("=" * 60)
    
    def print_success(self, message: str):
        """Print a success message."""
        if RICH_AVAILABLE and self.console:
            self.console.print(f"✅ {message}", style="bold green")
        elif COLORAMA_AVAILABLE:
            print(f"{Fore.GREEN}✅ {message}{Style.RESET_ALL}")
        else:
            print(f"✅ {message}")
    
    def print_error(self, message: str):
        """Print an error message."""
        if RICH_AVAILABLE and self.console:
            self.console.print(f"❌ {message}", style="bold red")
        elif COLORAMA_AVAILABLE:
            print(f"{Fore.RED}❌ {message}{Style.RESET_ALL}")
        else:
            print(f"❌ {message}")
    
    def print_warning(self, message: str):
        """Print a warning message."""
        if RICH_AVAILABLE and self.console:
            self.console.print(f"⚠️ {message}", style="bold yellow")
        elif COLORAMA_AVAILABLE:
            print(f"{Fore.YELLOW}⚠️ {message}{Style.RESET_ALL}")
        else:
            print(f"⚠️ {message}")
    
    def print_info(self, message: str):
        """Print an info message."""
        if RICH_AVAILABLE and self.console:
            self.console.print(f"ℹ️ {message}", style="cyan")
        elif COLORAMA_AVAILABLE:
            print(f"{Fore.CYAN}ℹ️ {message}{Style.RESET_ALL}")
        else:
            print(f"ℹ️ {message}")
    
    def format_transcription_table(self, transcriptions: List[str]) -> Optional[Table]:
        """Create a table for displaying transcriptions."""
        if not RICH_AVAILABLE:
            return None
            
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
            if RICH_AVAILABLE and self.console:
                self.console.print(test_text)
            else:
                print(test_text)
            return True
        except UnicodeEncodeError:
            return False
    
    def print_rtl_configuration_info(self):
        """Print RTL configuration information."""
        if RTL_ENHANCED_AVAILABLE:
            enhanced_rtl_display.print_configuration_info()
        else:
            print("\n⚠️ Enhanced RTL support not available")
            print("💡 For better Persian text display, install:")
            print("   pip install python-bidi arabic-reshaper")


# Global instance for easy access
terminal_display = TerminalDisplay()
