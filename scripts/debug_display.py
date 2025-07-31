#!/usr/bin/env python3
"""
Debug script to test Persian text display in different ways.
"""

import sys
import os

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_basic_display():
    """Test basic Persian text display."""
    print("=" * 60)
    print("🔍 BASIC DISPLAY TEST")
    print("=" * 60)
    
    persian_text = "سلام دنیا، این یک تست است"
    print(f"Basic print: {persian_text}")
    print(f"Raw text: {repr(persian_text)}")
    print(f"Length: {len(persian_text)}")
    print(f"Bytes: {persian_text.encode('utf-8')}")

def test_rich_display():
    """Test rich library display."""
    print("\n" + "=" * 60)
    print("🎨 RICH DISPLAY TEST")
    print("=" * 60)
    
    try:
        from rich.console import Console
        from rich.text import Text
        from rich.panel import Panel
        
        console = Console()
        persian_text = "سلام دنیا، این یک تست است"
        
        # Test 1: Direct console print
        console.print(f"Rich console: {persian_text}")
        
        # Test 2: Rich text
        rich_text = Text()
        rich_text.append("Rich text: ", style="bold")
        rich_text.append(persian_text, style="green")
        console.print(rich_text)
        
        # Test 3: Panel
        panel = Panel(persian_text, title="Persian Test", border_style="blue")
        console.print(panel)
        
    except ImportError as e:
        print(f"❌ Rich not available: {e}")
    except Exception as e:
        print(f"❌ Rich display error: {e}")

def test_terminal_display():
    """Test our custom terminal display."""
    print("\n" + "=" * 60)
    print("🖥️ TERMINAL DISPLAY TEST")
    print("=" * 60)
    
    try:
        from src.utils.terminal_display import terminal_display
        
        persian_text = "سلام دنیا، این یک تست است"
        
        # Test our custom display
        terminal_display.print_persian_preview(persian_text, 1)
        terminal_display.print_simple_preview(persian_text, 2)
        
    except ImportError as e:
        print(f"❌ Terminal display not available: {e}")
    except Exception as e:
        print(f"❌ Terminal display error: {e}")

def test_encoding():
    """Test encoding issues."""
    print("\n" + "=" * 60)
    print("🔤 ENCODING TEST")
    print("=" * 60)
    
    persian_text = "سلام دنیا، این یک تست است"
    
    print(f"System encoding: {sys.getdefaultencoding()}")
    print(f"Stdout encoding: {sys.stdout.encoding}")
    print(f"Stderr encoding: {sys.stderr.encoding}")
    
    # Test different encodings
    encodings = ['utf-8', 'utf-16', 'ascii', 'latin-1']
    
    for encoding in encodings:
        try:
            encoded = persian_text.encode(encoding)
            decoded = encoded.decode(encoding)
            print(f"{encoding}: {decoded} (✓)")
        except UnicodeEncodeError:
            print(f"{encoding}: Cannot encode (✗)")
        except UnicodeDecodeError:
            print(f"{encoding}: Cannot decode (✗)")

def test_whisper_output():
    """Test what Whisper might output."""
    print("\n" + "=" * 60)
    print("🎙️ WHISPER OUTPUT SIMULATION")
    print("=" * 60)
    
    # Simulate what Whisper might output
    whisper_outputs = [
        "سلام دنیا، این یک تست است",
        "این ایشه ایتا بیشه ایتا بیس آمری کنکشان آم چیزیگی در مدیب احسن بیککیشه می کن بیرم آمی",
        "از که این این تنوه بین از این محلون این این سوه این دیالی ده از تشمیخیو اون کم هستی نار پر اشمیم بین"
    ]
    
    for i, text in enumerate(whisper_outputs, 1):
        print(f"Whisper output {i}: {text}")
        print(f"Length: {len(text)}")
        print(f"Contains Persian: {'سلام' in text or 'دنیا' in text}")
        print("-" * 40)

def main():
    """Run all display tests."""
    print("🎙️ PERSIAN TEXT DISPLAY DEBUG")
    print("=" * 60)
    
    test_basic_display()
    test_rich_display()
    test_terminal_display()
    test_encoding()
    test_whisper_output()
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    print("If you see garbled text above, it might be due to:")
    print("1. Terminal font not supporting Persian characters")
    print("2. Terminal encoding issues")
    print("3. System locale configuration")
    print("\nTry these solutions:")
    print("1. Change your terminal font to one that supports Persian")
    print("2. Set LANG=en_US.UTF-8 in your shell")
    print("3. Use a different terminal application")

if __name__ == "__main__":
    main() 