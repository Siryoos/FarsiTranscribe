"""
Enhanced terminal display system with proper RTL (Right-to-Left) support for Persian/Farsi text.
Optimized architecture with fallback mechanisms and performance considerations.
"""

import os
import sys
import shutil
import unicodedata
from typing import List, Optional, Dict, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import re

try:
    from arabic_reshaper import reshape
    from bidi.algorithm import get_display
    RTL_SUPPORT_AVAILABLE = True
except ImportError:
    RTL_SUPPORT_AVAILABLE = False

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


class DisplayMode(Enum):
    """Available display modes for terminal output."""
    RTL_RICH = "rtl_rich"           # Full RTL support with Rich formatting
    RTL_PLAIN = "rtl_plain"         # RTL support with plain text
    LTR_RICH = "ltr_rich"           # Left-to-right with Rich formatting
    LTR_PLAIN = "ltr_plain"         # Basic left-to-right plain text
    FALLBACK = "fallback"           # Minimal fallback mode


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
    """High-performance RTL text processing with caching."""
    
    def __init__(self):
        self._cache: Dict[str, str] = {}
        self._cache_size_limit = 1000
        
    def process_persian_text(self, text: str) -> str:
        """Process Persian text for proper RTL display with caching."""
        if not text or not RTL_SUPPORT_AVAILABLE:
            return text
            
        # Check cache first for performance
        if text in self._cache:
            return self._cache[text]
            
        try:
            # Step 1: Reshape Arabic/Persian characters for proper joining
            reshaped_text = reshape(text)
            
            # Step 2: Apply bidirectional algorithm for proper RTL layout
            display_text = get_display(reshaped_text)
            
            # Cache the result
            self._manage_cache(text, display_text)
            
            return display_text
            
        except Exception as e:
            # Fallback to original text
            return text
    
    def _manage_cache(self, key: str, value: str):
        """Manage cache size to prevent memory issues."""
        if len(self._cache) >= self._cache_size_limit:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self._cache.keys())[:100]
            for old_key in oldest_keys:
                del self._cache[old_key]
        
        self._cache[key] = value


class TerminalCapabilityDetector:
    """Detect terminal capabilities for optimal display configuration."""
    
    @staticmethod
    def detect_capabilities() -> TerminalCapabilities:
        """Comprehensive terminal capability detection."""
        return TerminalCapabilities(
            unicode_support=TerminalCapabilityDetector._test_unicode_support(),
            rtl_support=TerminalCapabilityDetector._test_rtl_support(),
            color_support=TerminalCapabilityDetector._test_color_support(),
            width=TerminalCapabilityDetector._get_terminal_width(),
            rich_available=RICH_AVAILABLE,
            rtl_libs_available=RTL_SUPPORT_AVAILABLE
        )
    
    @staticmethod
    def _test_unicode_support() -> bool:
        """Test if terminal supports Unicode characters."""
        try:
            # Test with Persian characters
            test_chars = "سلام تست ۱۲۳"
            sys.stdout.write(test_chars)
            sys.stdout.flush()
            return True
        except UnicodeEncodeError:
            return False
    
    @staticmethod
    def _test_rtl_support() -> bool:
        """Test RTL library availability and functionality."""
        if not RTL_SUPPORT_AVAILABLE:
            return False
        
        try:
            # Test basic RTL processing
            test_text = "سلام دنیا"
            reshaped = reshape(test_text)
            displayed = get_display(reshaped)
            return True
        except Exception:
            return False
    
    @staticmethod
    def _test_color_support() -> bool:
        """Test terminal color support."""
        return (
            COLORAMA_AVAILABLE and 
            (os.getenv('TERM', '').lower() not in ['dumb', ''] or 
             os.getenv('COLORTERM', '') != '')
        )
    
    @staticmethod
    def _get_terminal_width() -> int:
        """Get terminal width with fallback."""
        try:
            return shutil.get_terminal_size().columns
        except Exception:
            return 80  # Fallback width


class EnhancedRTLTerminalDisplay:
    """Enhanced terminal display with comprehensive RTL support and performance optimization."""
    
    def __init__(self):
        self.capabilities = TerminalCapabilityDetector.detect_capabilities()
        self.rtl_processor = RTLTextProcessor()
        self.display_mode = self._determine_optimal_display_mode()
        
        # Initialize components based on capabilities
        if self.capabilities.rich_available and self.display_mode in [DisplayMode.RTL_RICH, DisplayMode.LTR_RICH]:
            self.console = Console(force_terminal=True, width=self.capabilities.width)
        else:
            self.console = None
            
        self._setup_unicode_environment()
    
    def _determine_optimal_display_mode(self) -> DisplayMode:
        """Determine the best display mode based on terminal capabilities."""
        if self.capabilities.rtl_libs_available and self.capabilities.unicode_support:
            if self.capabilities.rich_available:
                return DisplayMode.RTL_RICH
            else:
                return DisplayMode.RTL_PLAIN
        elif self.capabilities.rich_available:
            return DisplayMode.LTR_RICH
        elif self.capabilities.unicode_support:
            return DisplayMode.LTR_PLAIN
        else:
            return DisplayMode.FALLBACK
    
    def _setup_unicode_environment(self):
        """Configure environment for optimal Unicode support."""
        # Set environment variables
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        os.environ['LC_ALL'] = 'en_US.UTF-8'
        
        # Reconfigure stdout/stderr if possible
        if hasattr(sys.stdout, 'reconfigure'):
            try:
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass
    
    def print_persian_preview(self, text: str, part_number: int, max_length: int = 100):
        """Print Persian text preview with optimal formatting based on capabilities."""
        if not text or not text.strip():
            return
        
        # Process text based on display mode
        processed_text = self._process_text_for_display(text, max_length)
        
        if self.display_mode == DisplayMode.RTL_RICH:
            self._print_rtl_rich_preview(processed_text, part_number)
        elif self.display_mode == DisplayMode.RTL_PLAIN:
            self._print_rtl_plain_preview(processed_text, part_number)
        elif self.display_mode == DisplayMode.LTR_RICH:
            self._print_ltr_rich_preview(processed_text, part_number)
        elif self.display_mode == DisplayMode.LTR_PLAIN:
            self._print_ltr_plain_preview(processed_text, part_number)
        else:
            self._print_fallback_preview(processed_text, part_number)
    
    def _process_text_for_display(self, text: str, max_length: int) -> str:
        """Process text for display with length management."""
        cleaned_text = text.strip()
        
        # Truncate if necessary
        if len(cleaned_text) > max_length:
            cleaned_text = cleaned_text[:max_length] + "..."
        
        # Apply RTL processing if available
        if self.display_mode in [DisplayMode.RTL_RICH, DisplayMode.RTL_PLAIN]:
            return self.rtl_processor.process_persian_text(cleaned_text)
        
        return cleaned_text
    
    def _print_rtl_rich_preview(self, text: str, part_number: int):
        """Print with Rich formatting and RTL support."""
        try:
            # Create preview box with proper RTL text
            preview_text = f"Part {part_number} Preview: {text}"
            
            # Use Rich panel for better display
            panel = Panel(
                preview_text,
                title="🎙️ Transcription Preview",
                border_style="cyan",
                padding=(0, 1),
                width=min(self.capabilities.width - 4, 120)
            )
            
            self.console.print(panel)
            
        except Exception as e:
            # Fallback to plain RTL
            self._print_rtl_plain_preview(text, part_number)
    
    def _print_rtl_plain_preview(self, text: str, part_number: int):
        """Print with RTL support but plain formatting."""
        try:
            if COLORAMA_AVAILABLE:
                print(f"{Fore.CYAN}╭{'─' * (self.capabilities.width - 10)}╮{Style.RESET_ALL}")
                print(f"{Fore.CYAN}│ 🎙️ Transcription Preview{' ' * (self.capabilities.width - 35)}│{Style.RESET_ALL}")
                print(f"{Fore.CYAN}├{'─' * (self.capabilities.width - 10)}┤{Style.RESET_ALL}")
                print(f"{Fore.WHITE}│ Part {part_number} Preview: {text}{' ' * max(0, self.capabilities.width - len(text) - 25)}│{Style.RESET_ALL}")
                print(f"{Fore.CYAN}╰{'─' * (self.capabilities.width - 10)}╯{Style.RESET_ALL}")
            else:
                print("─" * 50)
                print(f"Part {part_number} Preview: {text}")
                print("─" * 50)
                
        except Exception:
            self._print_fallback_preview(text, part_number)
    
    def _print_ltr_rich_preview(self, text: str, part_number: int):
        """Print with Rich formatting but LTR text."""
        try:
            rich_text = Text()
            rich_text.append(f"Part {part_number} Preview: ", style="bold cyan")
            rich_text.append(text, style="white")
            
            panel = Panel(
                rich_text,
                title="🎙️ Transcription Preview",
                border_style="blue",
                padding=(0, 1)
            )
            
            self.console.print(panel)
            
        except Exception:
            self._print_ltr_plain_preview(text, part_number)
    
    def _print_ltr_plain_preview(self, text: str, part_number: int):
        """Print with plain formatting and LTR text."""
        try:
            if COLORAMA_AVAILABLE:
                print(f"{Fore.CYAN}Part {part_number} Preview:{Style.RESET_ALL} {Fore.WHITE}{text}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{'-' * 50}{Style.RESET_ALL}")
            else:
                print(f"Part {part_number} Preview: {text}")
                print("-" * 50)
                
        except Exception:
            self._print_fallback_preview(text, part_number)
    
    def _print_fallback_preview(self, text: str, part_number: int):
        """Minimal fallback preview for maximum compatibility."""
        try:
            print(f"Part {part_number}: {text}")
        except UnicodeEncodeError:
            # Ultimate fallback - encode to ASCII with replacement
            safe_text = text.encode('ascii', errors='replace').decode('ascii')
            print(f"Part {part_number}: {safe_text}")
    
    def print_simple_preview(self, text: str, part_number: str):
        """Simple preview for secondary content."""
        if not text or not text.strip():
            return
        
        processed_text = self._process_text_for_display(text, 150)
        
        try:
            if COLORAMA_AVAILABLE and self.capabilities.color_support:
                print(f"{Fore.YELLOW}Part {part_number}.2 Preview:{Style.RESET_ALL}")
                print(f"{processed_text}")
                print(f"{Fore.YELLOW}{'-' * 50}{Style.RESET_ALL}")
            else:
                print(f"Part {part_number}.2 Preview:")
                print(processed_text)
                print("-" * 50)
                
        except Exception:
            self._print_fallback_preview(processed_text, part_number)
    
    def print_configuration_info(self):
        """Print terminal configuration information for debugging."""
        print("\n🔧 Terminal Configuration:")
        print(f"   • Display Mode: {self.display_mode.value}")
        print(f"   • Unicode Support: {'✅' if self.capabilities.unicode_support else '❌'}")
        print(f"   • RTL Support: {'✅' if self.capabilities.rtl_support else '❌'}")
        print(f"   • Color Support: {'✅' if self.capabilities.color_support else '❌'}")
        print(f"   • Rich Available: {'✅' if self.capabilities.rich_available else '❌'}")
        print(f"   • Terminal Width: {self.capabilities.width}")
        print(f"   • RTL Libraries: {'✅' if self.capabilities.rtl_libs_available else '❌'}")
        
        if not self.capabilities.rtl_libs_available:
            print("\n💡 For better Persian text display, install RTL libraries:")
            print("   pip install python-bidi arabic-reshaper")


# Global instance
enhanced_rtl_display = EnhancedRTLTerminalDisplay()
