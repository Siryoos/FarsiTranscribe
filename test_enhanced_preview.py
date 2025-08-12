#!/usr/bin/env python3
"""
Test script for the enhanced preview display system.
"""

import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.enhanced_preview_display import create_preview_display


def test_enhanced_preview():
    """Test the enhanced preview display system."""
    print("🧪 Testing Enhanced Preview Display System")
    print("=" * 50)
    
    # Create preview display
    total_chunks = 100
    estimated_duration = 300  # 5 minutes
    
    preview = create_preview_display(total_chunks, estimated_duration)
    
    print(f"✅ Created preview display for {total_chunks} chunks")
    print(f"⏱️  Estimated duration: {estimated_duration} seconds")
    print()
    
    # Simulate transcription process
    print("🔄 Simulating transcription process...")
    print("   (Press Ctrl+C to stop)")
    
    try:
        with preview:
            # Simulate chunks being processed
            for chunk_id in range(total_chunks):
                # Add chunk
                chunk_start = (chunk_id / total_chunks) * estimated_duration
                chunk_end = ((chunk_id + 1) / total_chunks) * estimated_duration
                chunk_duration = chunk_end - chunk_start
                
                preview.add_chunk(chunk_id, chunk_start, chunk_end, chunk_duration)
                
                # Simulate transcription progress
                for progress in [0, 25, 50, 75, 100]:
                    preview.update_chunk_progress(chunk_id, progress)
                    time.sleep(0.1)  # Small delay to see progress
                
                # Add some transcribed text
                sample_texts = [
                    "ن در قرب و ا ب دوخ را از طرح م",
                    "ہی اب ال ۱۳ ۱۰ دن ایمن کری ب ردی اق آ درج می امتزر و در هرا چی ماست این آز ب ن م",
                    "ن امر ای ہی توب؟ در این دایزین احى علق ، ازی انوب",
                    "سلام، چطور هستید؟",
                    "امروز هوا خوب است."
                ]
                
                sample_text = sample_texts[chunk_id % len(sample_texts)]
                preview.update_chunk_progress(chunk_id, 100, sample_text)
                
                # Small delay between chunks
                time.sleep(0.2)
                
    except KeyboardInterrupt:
        print("\n⏹️  Test stopped by user")
    
    print("\n✅ Test completed!")


def test_simple_preview():
    """Test the simple preview display system."""
    print("🧪 Testing Simple Preview Display System")
    print("=" * 50)
    
    # Create simple preview display
    total_chunks = 20
    preview = create_preview_display(total_chunks, use_enhanced=False)
    
    print(f"✅ Created simple preview display for {total_chunks} chunks")
    print()
    
    # Simulate transcription process
    print("🔄 Simulating transcription process...")
    print("   (Press Ctrl+C to stop)")
    
    try:
        # Simulate chunks being processed
        for chunk_id in range(total_chunks):
            # Add chunk
            chunk_start = (chunk_id / total_chunks) * 60  # 1 minute total
            chunk_end = ((chunk_id + 1) / total_chunks) * 60
            chunk_duration = chunk_end - chunk_start
            
            preview.add_chunk(chunk_id, chunk_start, chunk_end, chunk_duration)
            
            # Simulate transcription progress
            for progress in [0, 50, 100]:
                preview.update_chunk_progress(chunk_id, progress)
                time.sleep(0.1)
            
            # Add some transcribed text
            sample_text = f"Sample text for chunk {chunk_id + 1}"
            preview.update_chunk_progress(chunk_id, 100, sample_text)
            
            # Display current state
            preview.display()
            
            # Small delay between chunks
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n⏹️  Test stopped by user")
    
    print("\n✅ Test completed!")


def main():
    """Run the tests."""
    print("🎙️  Enhanced Preview Display Test Suite")
    print("=" * 60)
    
    # Check if Rich is available
    try:
        import rich
        print("✅ Rich library available - enhanced preview will be used")
        use_enhanced = True
    except ImportError:
        print("⚠️  Rich library not available - simple preview will be used")
        use_enhanced = False
    
    print()
    
    if use_enhanced:
        test_enhanced_preview()
    else:
        test_simple_preview()
    
    print("\n📚 Next steps:")
    print("1. Install Rich library: pip install rich")
    print("2. Run the main transcription: python main.py <audio_file>")
    print("3. The enhanced preview will automatically activate")


if __name__ == "__main__":
    main()
