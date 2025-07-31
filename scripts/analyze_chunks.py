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
from src.utils.chunk_calculator import create_chunk_calculator
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
        
        # Create unified chunk calculator
        chunk_calculator = create_chunk_calculator(config)
        
        # Get comprehensive chunk analysis
        chunk_analysis = chunk_calculator.get_chunk_analysis(duration_ms)
        num_chunks = chunk_analysis['total_chunks']
        
        # Get memory and time estimates
        memory_estimate = chunk_calculator.estimate_memory_usage(
            duration_ms, 
            audio.frame_rate, 
            audio.channels
        )
        time_estimate = chunk_calculator.estimate_processing_time(duration_ms, config.model_name)
        
        print(f"📦 Chunk Analysis:")
        print(f"   Total Chunks: {num_chunks}")
        print(f"   Chunk Duration: {chunk_analysis['chunk_duration_ms']}ms ({chunk_analysis['chunk_duration_ms']/1000:.1f}s)")
        print(f"   Overlap: {chunk_analysis['overlap_ms']}ms ({chunk_analysis['overlap_ms']/1000:.1f}s)")
        print(f"   Effective Chunk Duration: {chunk_analysis['effective_chunk_duration_ms']}ms ({chunk_analysis['effective_chunk_duration_ms']/1000:.1f}s)")
        print(f"   Average Chunk Duration: {chunk_analysis['avg_chunk_duration_ms']:.0f}ms")
        print()
        
        # Show chunk details
        print(f"🔢 Chunk Details:")
        for chunk_info in chunk_analysis['chunk_info']:
            print(f"   Chunk {chunk_info.index+1:2d}: {chunk_info.start_ms:6.0f}ms - {chunk_info.end_ms:6.0f}ms ({chunk_info.duration_ms:6.0f}ms)")
        
        print()
        
        print(f"💾 Memory Estimation:")
        print(f"   Memory per chunk: ~{memory_estimate['memory_per_chunk_mb']:.1f} MB")
        print(f"   Peak memory usage: ~{memory_estimate['peak_memory_mb']:.1f} MB")
        print(f"   Total memory for all chunks: ~{memory_estimate['total_memory_mb']:.1f} MB")
        print(f"   Max concurrent chunks: {memory_estimate['max_concurrent_chunks']}")
        print(f"   Processing mode: {'Streaming' if config.memory_efficient_mode else 'Batch'}")
        
        print(f"⏱️  Time Estimation:")
        print(f"   Time per chunk: ~{time_estimate['time_per_chunk_seconds']:.0f} seconds")
        print(f"   Total processing time: ~{time_estimate['total_time_seconds']:.0f} seconds ({time_estimate['total_time_minutes']:.1f} minutes)")
        print(f"   Chunks per minute: ~{time_estimate['chunks_per_minute']:.1f}")
        
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
            'chunk_analysis': chunk_analysis,
            'memory_estimate': memory_estimate,
            'time_estimate': time_estimate
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
        
        # Use unified chunk calculator
        chunk_calculator = create_chunk_calculator(config)
        audio = AudioSegment.from_file(audio_file_path)
        duration_ms = len(audio)
        
        chunk_analysis = chunk_calculator.get_chunk_analysis(duration_ms)
        num_chunks = chunk_analysis['total_chunks']
        
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