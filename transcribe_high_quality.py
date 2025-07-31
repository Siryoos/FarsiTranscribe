#!/usr/bin/env python3
"""
High-quality transcription script with enhanced anti-repetition features.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import UnifiedAudioTranscriber
from src.core.config import TranscriptionConfig


def create_high_quality_config(audio_file_path: str) -> TranscriptionConfig:
    """Create optimized configuration for high-quality transcription."""
    
    base_filename = Path(audio_file_path).stem
    output_dir = f"./output/{base_filename}_high_quality"
    
    config = TranscriptionConfig(
        # Use large-v3 model for best quality
        model_name="large-v3",
        language="fa",
        
        # Optimal chunk settings for Persian audio
        chunk_duration_ms=20000,  # 20 seconds
        overlap_ms=500,  # 500ms overlap to catch transitions
        
        # Enhanced anti-repetition settings
        repetition_threshold=0.75,  # Stricter threshold
        max_word_repetition=1,  # Very strict repetition limit
        
        # Quality settings
        temperature=0.0,  # Deterministic output
        condition_on_previous_text=False,  # Prevent context contamination
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.0,  # Stricter compression ratio
        
        # Processing settings
        batch_size=2,  # Smaller batches for better quality
        enable_sentence_preview=True,
        preview_sentence_count=3,
        
        # Output settings
        output_directory=output_dir,
        save_individual_parts=False,
        unified_filename_suffix="_unified_transcription.txt"
    )
    
    return config


def transcribe_with_quality_check(audio_file_path: str):
    """Transcribe audio with quality checks and anti-repetition."""
    
    if not os.path.exists(audio_file_path):
        print(f"❌ Error: Audio file not found: {audio_file_path}")
        return
    
    print("🎙️  High-Quality Transcription with Anti-Repetition")
    print("=" * 60)
    print(f"📁 Audio File: {audio_file_path}")
    print("🔧 Configuration:")
    print("   • Model: large-v3 (highest quality)")
    print("   • Anti-repetition: Enhanced (strict)")
    print("   • Chunk size: 20 seconds")
    print("   • Overlap: 500ms")
    print("=" * 60)
    
    # Create configuration
    config = create_high_quality_config(audio_file_path)
    
    # Ensure output directory exists
    os.makedirs(config.output_directory, exist_ok=True)
    
    try:
        # Run transcription
        with UnifiedAudioTranscriber(config) as transcriber:
            transcription = transcriber.transcribe_file(audio_file_path)
            
            if transcription:
                print("\n✅ Transcription completed successfully!")
                print(f"📄 Output directory: {config.output_directory}")
                
                # Show quality metrics
                original_file = Path(config.output_directory) / f"{Path(audio_file_path).stem}_unified_transcription.txt"
                cleaned_file = Path(config.output_directory) / f"{Path(audio_file_path).stem}_cleaned_transcription.txt"
                
                if original_file.exists() and cleaned_file.exists():
                    with open(original_file, 'r', encoding='utf-8') as f:
                        original_text = f.read()
                    with open(cleaned_file, 'r', encoding='utf-8') as f:
                        cleaned_text = f.read()
                    
                    reduction = ((len(original_text) - len(cleaned_text)) / len(original_text) * 100) if original_text else 0
                    
                    print(f"\n📊 Quality Metrics:")
                    print(f"   • Original: {len(original_text):,} characters")
                    print(f"   • Cleaned: {len(cleaned_text):,} characters")
                    print(f"   • Repetition removed: {reduction:.1f}%")
                
                # Show preview
                preview_length = 500
                if len(transcription) > preview_length:
                    preview = transcription[:preview_length] + "..."
                else:
                    preview = transcription
                
                print(f"\n📖 Preview:")
                print("-" * 40)
                print(preview)
                print("-" * 40)
                
                return transcription
            else:
                print("⚠️  Transcription completed but no text was generated.")
                return None
                
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python transcribe_high_quality.py <audio_file>")
        print("\nExample:")
        print("  python transcribe_high_quality.py jalase_bi_va_zirsakht.m4a")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    transcribe_with_quality_check(audio_file)


if __name__ == "__main__":
    main()