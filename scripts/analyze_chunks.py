#!/usr/bin/env python3
"""
Chunk analysis script for FarsiTranscribe.
Shows detailed information about how audio will be chunked for processing.
"""

import os
import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import ConfigFactory
from src.utils.audio_preprocessor import AudioPreprocessor
from pydub import AudioSegment


def analyze_audio_chunks(audio_file_path: str, config_name: str = "memory-optimized"):
    """Analyze how an audio file will be chunked for processing."""
    
    print(f"🔍 Analyzing chunks for: {audio_file_path}")
    print("=" * 60)
    
    # Create configuration
    if config_name == "memory-optimized":
        config = ConfigFactory.create_memory_optimized_config()
    elif config_name == "cpu-optimized":
        config = ConfigFactory.create_cpu_optimized_config()
    elif config_name == "balanced":
        config = ConfigFactory.create_optimized_config()
    elif config_name == "high":
        config = ConfigFactory.create_high_quality_config()
    else:
        config = ConfigFactory.create_optimized_config()
    
    # Display configuration
    print(f"📊 Configuration: {config_name}")
    print(f"   Model: {config.model_name}")
    print(f"   Chunk Duration: {config.chunk_duration_ms}ms ({config.chunk_duration_ms/1000:.1f}s)")
    print(f"   Overlap: {config.overlap_ms}ms ({config.overlap_ms/1000:.1f}s)")
    print(f"   Memory Efficient: {config.memory_efficient_mode}")
    print()
    
    # Load audio file
    try:
        audio = AudioSegment.from_file(audio_file_path)
        duration_ms = len(audio)
        duration_seconds = duration_ms / 1000
        
        print(f"🎵 Audio File Information:")
        print(f"   Duration: {duration_seconds:.1f} seconds ({duration_ms}ms)")
        print(f"   Sample Rate: {audio.frame_rate} Hz")
        print(f"   Channels: {audio.channels}")
        print(f"   File Size: {os.path.getsize(audio_file_path) / (1024*1024):.1f} MB")
        print()
        
        # Calculate chunks
        chunk_duration = config.chunk_duration_ms
        overlap = config.overlap_ms
        effective_chunk_duration = chunk_duration - overlap
        
        # Calculate number of chunks
        if duration_ms <= chunk_duration:
            num_chunks = 1
        else:
            num_chunks = int((duration_ms - overlap) / effective_chunk_duration) + 1
        
        print(f"📦 Chunk Analysis:")
        print(f"   Total Chunks: {num_chunks}")
        print(f"   Chunk Duration: {chunk_duration}ms ({chunk_duration/1000:.1f}s)")
        print(f"   Overlap: {overlap}ms ({overlap/1000:.1f}s)")
        print(f"   Effective Chunk Duration: {effective_chunk_duration}ms ({effective_chunk_duration/1000:.1f}s)")
        print()
        
        # Show chunk details
        print(f"🔢 Chunk Details:")
        for i in range(num_chunks):
            start_ms = i * effective_chunk_duration
            end_ms = min(start_ms + chunk_duration, duration_ms)
            chunk_duration_actual = end_ms - start_ms
            
            print(f"   Chunk {i+1:2d}: {start_ms:6.0f}ms - {end_ms:6.0f}ms ({chunk_duration_actual:6.0f}ms)")
        
        print()
        
        # Memory estimation
        estimated_memory_per_chunk = chunk_duration * audio.frame_rate * audio.channels * 2 / (1024 * 1024)  # MB
        total_memory_estimate = estimated_memory_per_chunk * min(num_chunks, 3)  # Assume max 3 chunks in memory
        
        print(f"💾 Memory Estimation:")
        print(f"   Memory per chunk: ~{estimated_memory_per_chunk:.1f} MB")
        print(f"   Peak memory usage: ~{total_memory_estimate:.1f} MB")
        print(f"   Processing mode: {'Streaming' if config.memory_efficient_mode else 'Batch'}")
        
        # Processing time estimation
        if config.model_name == "tiny":
            time_per_chunk = 2  # seconds
        elif config.model_name == "base":
            time_per_chunk = 4
        elif config.model_name == "small":
            time_per_chunk = 8
        elif config.model_name == "medium":
            time_per_chunk = 15
        else:  # large models
            time_per_chunk = 30
        
        estimated_total_time = num_chunks * time_per_chunk
        print(f"   Estimated processing time: ~{estimated_total_time:.0f} seconds")
        
        print()
        
        # Recommendations
        print(f"💡 Recommendations:")
        if num_chunks > 20:
            print(f"   ⚠️  Large file detected ({num_chunks} chunks)")
            print(f"   💾 Use --quality memory-optimized for better memory management")
        elif num_chunks > 10:
            print(f"   📊 Medium file ({num_chunks} chunks)")
            print(f"   ⚡ Use --quality cpu-optimized for balanced performance")
        else:
            print(f"   ✅ Small file ({num_chunks} chunks)")
            print(f"   🎯 Use --quality balanced for best quality")
        
        if config.memory_efficient_mode:
            print(f"   🔄 Streaming mode will be used for memory efficiency")
        
        return {
            'num_chunks': num_chunks,
            'duration_seconds': duration_seconds,
            'chunk_duration_ms': chunk_duration,
            'overlap_ms': overlap,
            'estimated_memory_mb': total_memory_estimate,
            'estimated_time_seconds': estimated_total_time
        }
        
    except Exception as e:
        print(f"❌ Error analyzing audio file: {e}")
        return None


def compare_configurations(audio_file_path: str):
    """Compare chunk analysis across different configurations."""
    
    print(f"🔍 Comparing configurations for: {audio_file_path}")
    print("=" * 80)
    
    configs = {
        "memory-optimized": ConfigFactory.create_memory_optimized_config(),
        "cpu-optimized": ConfigFactory.create_cpu_optimized_config(),
        "balanced": ConfigFactory.create_optimized_config(),
        "high": ConfigFactory.create_high_quality_config()
    }
    
    results = {}
    
    for name, config in configs.items():
        print(f"\n📊 {name.upper()} Configuration:")
        print(f"   Model: {config.model_name}")
        print(f"   Chunk Duration: {config.chunk_duration_ms}ms")
        print(f"   Overlap: {config.overlap_ms}ms")
        
        # Calculate chunks
        audio = AudioSegment.from_file(audio_file_path)
        duration_ms = len(audio)
        chunk_duration = config.chunk_duration_ms
        overlap = config.overlap_ms
        effective_chunk_duration = chunk_duration - overlap
        
        if duration_ms <= chunk_duration:
            num_chunks = 1
        else:
            num_chunks = int((duration_ms - overlap) / effective_chunk_duration) + 1
        
        results[name] = {
            'num_chunks': num_chunks,
            'model': config.model_name,
            'chunk_duration': chunk_duration,
            'memory_efficient': config.memory_efficient_mode
        }
        
        print(f"   Chunks: {num_chunks}")
        print(f"   Memory Efficient: {config.memory_efficient_mode}")
    
    # Summary table
    print(f"\n📋 Summary Comparison:")
    print(f"{'Configuration':<20} {'Model':<8} {'Chunks':<8} {'Chunk Duration':<15} {'Memory Efficient':<15}")
    print("-" * 80)
    
    for name, result in results.items():
        print(f"{name:<20} {result['model']:<8} {result['num_chunks']:<8} {result['chunk_duration']:<15} {result['memory_efficient']:<15}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze audio chunking for FarsiTranscribe")
    parser.add_argument("audio_file", help="Path to the audio file to analyze")
    parser.add_argument(
        "--config", 
        choices=["memory-optimized", "cpu-optimized", "balanced", "high"],
        default="memory-optimized",
        help="Configuration to analyze (default: memory-optimized)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all configurations"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"❌ Error: Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    if args.compare:
        compare_configurations(args.audio_file)
    else:
        analyze_audio_chunks(args.audio_file, args.config)


if __name__ == "__main__":
    main() 