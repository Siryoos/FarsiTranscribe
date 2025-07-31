#!/usr/bin/env python3
"""
Test script for memory optimization features.
Demonstrates different memory usage patterns and optimization techniques.
"""

import os
import sys
import psutil
import time
import gc
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import ConfigFactory
from src.core.transcriber import MemoryManager, cleanup_shared_model


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def print_memory_info(label: str):
    """Print memory usage information."""
    memory_mb = get_memory_usage()
    print(f"📊 {label}: {memory_mb:.1f} MB")


def test_memory_manager():
    """Test the memory manager functionality."""
    print("🧪 Testing Memory Manager")
    print("=" * 50)
    
    # Create different configurations
    configs = {
        "memory-optimized": ConfigFactory.create_memory_optimized_config(),
        "cpu-optimized": ConfigFactory.create_cpu_optimized_config(),
        "balanced": ConfigFactory.create_optimized_config(),
        "high-quality": ConfigFactory.create_high_quality_config()
    }
    
    for name, config in configs.items():
        print(f"\n🔧 Testing {name} configuration:")
        print(f"   Model: {config.model_name}")
        print(f"   Device: {config.device}")
        print(f"   Chunk duration: {config.chunk_duration_ms}ms")
        print(f"   Memory threshold: {config.memory_threshold_mb}MB")
        print(f"   Cleanup interval: {config.cleanup_interval_seconds}s")
        print(f"   Memory efficient mode: {config.memory_efficient_mode}")
        print(f"   Enable preprocessing: {config.enable_preprocessing}")
        print(f"   Num workers: {config.num_workers}")
        
        # Create memory manager
        memory_manager = MemoryManager(config)
        
        # Test memory checking
        is_high = memory_manager.check_memory_usage()
        print(f"   Current memory usage high: {is_high}")
        
        # Test cleanup
        print("   Performing memory cleanup...")
        memory_manager.cleanup(force=True)
        
        print_memory_info(f"After {name} cleanup")


def test_model_cleanup():
    """Test model cleanup functionality."""
    print("\n🧪 Testing Model Cleanup")
    print("=" * 50)
    
    print_memory_info("Before model cleanup")
    
    # Simulate model cleanup
    cleanup_shared_model()
    gc.collect()
    
    print_memory_info("After model cleanup")


def test_configuration_comparison():
    """Compare memory usage of different configurations."""
    print("\n🧪 Configuration Memory Comparison")
    print("=" * 50)
    
    configs = {
        "memory-optimized": ConfigFactory.create_memory_optimized_config(),
        "cpu-optimized": ConfigFactory.create_cpu_optimized_config(),
        "balanced": ConfigFactory.create_optimized_config(),
        "high-quality": ConfigFactory.create_high_quality_config()
    }
    
    print("Configuration | Model | RAM Est. | Speed | Quality | Memory Efficient")
    print("-" * 80)
    
    for name, config in configs.items():
        # Estimate RAM usage based on model size
        model_ram = {
            "tiny": 39,
            "base": 74,
            "small": 244,
            "medium": 769,
            "large-v3": 1550
        }.get(config.model_name, 500)
        
        # Add overhead for processing
        total_ram = model_ram + (100 if config.enable_preprocessing else 50)
        
        speed = "Fast" if config.model_name in ["tiny", "base"] else "Medium" if config.model_name == "small" else "Slow"
        quality = "Low" if config.model_name in ["tiny", "base"] else "Good" if config.model_name == "small" else "High"
        
        print(f"{name:15} | {config.model_name:5} | {total_ram:6}MB | {speed:5} | {quality:6} | {config.memory_efficient_mode}")


def demonstrate_usage():
    """Demonstrate how to use memory optimization features."""
    print("\n📖 Memory Optimization Usage Examples")
    print("=" * 50)
    
    examples = [
        {
            "title": "Low RAM System (< 4GB)",
            "command": "python main.py audio.m4a --quality memory-optimized",
            "description": "Uses smallest model, minimal preprocessing, frequent cleanup"
        },
        {
            "title": "Medium RAM System (4-8GB)",
            "command": "python main.py audio.m4a --quality cpu-optimized",
            "description": "Balanced approach with medium model and moderate preprocessing"
        },
        {
            "title": "High RAM System (> 8GB)",
            "command": "python main.py audio.m4a --quality balanced",
            "description": "Best quality with large model and full preprocessing"
        },
        {
            "title": "Large Audio Files (> 100MB)",
            "command": "python main.py large_audio.m4a --quality memory-optimized",
            "description": "Automatically uses streaming mode for large files"
        }
    ]
    
    for example in examples:
        print(f"\n🎯 {example['title']}")
        print(f"   Command: {example['command']}")
        print(f"   Description: {example['description']}")


def main():
    """Main test function."""
    print("🧪 FarsiTranscribe Memory Optimization Test")
    print("=" * 60)
    
    print_memory_info("Initial memory usage")
    
    # Run tests
    test_memory_manager()
    test_model_cleanup()
    test_configuration_comparison()
    demonstrate_usage()
    
    print("\n" + "=" * 60)
    print("✅ Memory optimization test completed!")
    print("\n💡 Tips for optimal memory usage:")
    print("   1. Use --quality memory-optimized for low RAM systems")
    print("   2. Monitor memory usage during transcription")
    print("   3. Close other applications before processing large files")
    print("   4. Use CPU-only torch installation if GPU memory is limited")
    print("   5. Check MEMORY_OPTIMIZATION.md for detailed guide")
    
    print_memory_info("Final memory usage")


if __name__ == "__main__":
    main() 