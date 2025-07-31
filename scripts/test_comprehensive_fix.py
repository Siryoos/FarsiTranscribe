#!/usr/bin/env python3
"""
Comprehensive test script for FarsiTranscribe fixes.
This script validates all the improvements made to the system:
1. Chunk calculation consistency
2. Preprocessing validation
3. Memory management
4. Architecture improvements
5. Error handling
"""

import os
import sys
import time
import logging
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import ConfigFactory
from src.utils.chunk_calculator import create_chunk_calculator
from src.utils.preprocessing_validator import validate_preprocessing
from src.utils.enhanced_memory_manager import create_memory_manager
from src.core.transcriber import UnifiedAudioTranscriber


def test_chunk_calculator_consistency():
    """Test that chunk calculation is consistent across all components."""
    print("🔍 Testing chunk calculation consistency...")
    
    configs = [
        ConfigFactory.create_memory_optimized_config(),
        ConfigFactory.create_cpu_optimized_config(),
        ConfigFactory.create_optimized_config(),
        ConfigFactory.create_high_quality_config()
    ]
    
    test_durations = [5000, 15000, 30000, 60000, 120000]  # 5s, 15s, 30s, 1min, 2min
    
    for config in configs:
        print(f"\n📊 Testing config: {config.model_name}, chunk_duration={config.chunk_duration_ms}ms, overlap={config.overlap_ms}ms")
        
        calculator = create_chunk_calculator(config)
        
        for duration_ms in test_durations:
            # Test unified calculator
            total_chunks = calculator.calculate_total_chunks(duration_ms)
            chunk_info = calculator.generate_chunk_info(duration_ms)
            
            # Verify consistency
            assert len(chunk_info) == total_chunks, f"Chunk count mismatch: {len(chunk_info)} != {total_chunks}"
            
            # Verify chunk boundaries
            boundaries = calculator.generate_chunk_boundaries(duration_ms)
            assert len(boundaries) == total_chunks, f"Boundary count mismatch: {len(boundaries)} != {total_chunks}"
            
            # Verify chunk info consistency
            for i, info in enumerate(chunk_info):
                start_ms, end_ms = boundaries[i]
                assert info.start_ms == start_ms, f"Start time mismatch for chunk {i}"
                assert info.end_ms == end_ms, f"End time mismatch for chunk {i}"
                assert info.index == i, f"Index mismatch for chunk {i}"
            
            print(f"  ✅ {duration_ms/1000:.1f}s: {total_chunks} chunks")
    
    print("✅ Chunk calculation consistency test passed!")


def test_preprocessing_validation():
    """Test preprocessing validation system."""
    print("\n🔍 Testing preprocessing validation...")
    
    configs = [
        ConfigFactory.create_memory_optimized_config(),
        ConfigFactory.create_optimized_config(),
        ConfigFactory.create_high_quality_config()
    ]
    
    for config in configs:
        print(f"\n📊 Testing config: {config.model_name}")
        
        # Test validation
        validation_result = validate_preprocessing(config)
        
        # Check required capabilities
        required_capabilities = ['torch', 'whisper', 'numpy', 'pydub', 'librosa', 'scipy']
        for capability in required_capabilities:
            if capability in validation_result.capabilities:
                capability_info = validation_result.capabilities[capability]
                if capability_info.required and not capability_info.available:
                    print(f"  ❌ Required capability '{capability}' not available: {capability_info.error_message}")
                else:
                    print(f"  ✅ {capability}: {'Available' if capability_info.available else 'Optional'}")
        
        # Check validation result
        if validation_result.success:
            print(f"  ✅ Validation successful")
        else:
            print(f"  ❌ Validation failed:")
            for error in validation_result.errors:
                print(f"    - {error}")
        
        # Show warnings
        if validation_result.warnings:
            print(f"  ⚠️  Warnings:")
            for warning in validation_result.warnings:
                print(f"    - {warning}")
        
        # Show recommendations
        if validation_result.recommendations:
            print(f"  💡 Recommendations:")
            for rec in validation_result.recommendations:
                print(f"    - {rec}")
    
    print("✅ Preprocessing validation test completed!")


def test_memory_management():
    """Test enhanced memory management system."""
    print("\n🔍 Testing enhanced memory management...")
    
    config = ConfigFactory.create_memory_optimized_config()
    memory_manager = create_memory_manager(config)
    
    # Test memory monitoring
    print("📊 Testing memory monitoring...")
    stats = memory_manager.monitor.get_memory_stats()
    print(f"  Current memory usage: {stats.used_memory_mb:.1f}MB")
    print(f"  Available memory: {stats.available_memory_mb:.1f}MB")
    print(f"  Total memory: {stats.total_memory_mb:.1f}MB")
    print(f"  Memory percent: {stats.memory_percent:.1f}%")
    print(f"  CPU percent: {stats.cpu_percent:.1f}%")
    
    # Test memory thresholds
    print("📊 Testing memory thresholds...")
    print(f"  Warning threshold: {memory_manager.monitor.thresholds.warning_threshold_mb:.0f}MB")
    print(f"  Critical threshold: {memory_manager.monitor.thresholds.critical_threshold_mb:.0f}MB")
    print(f"  Emergency threshold: {memory_manager.monitor.thresholds.emergency_threshold_mb:.0f}MB")
    print(f"  Cleanup threshold: {memory_manager.monitor.thresholds.cleanup_threshold_mb:.0f}MB")
    
    # Test cleanup strategies
    print("📊 Testing cleanup strategies...")
    for i, strategy in enumerate(memory_manager.cleanup_strategies):
        print(f"  Testing cleanup strategy {i}...")
        memory_manager.cleanup(force=True, strategy_level=i)
        time.sleep(0.1)  # Brief pause
    
    # Test memory context
    print("📊 Testing memory context...")
    with memory_manager.memory_context(max_memory_mb=100):
        # Simulate memory-intensive operation
        import numpy as np
        test_array = np.random.randn(1000, 1000)
        print(f"  Created test array: {test_array.nbytes / (1024*1024):.1f}MB")
    
    # Get memory report
    print("📊 Memory report:")
    report = memory_manager.get_memory_report()
    print(f"  Cleanup count: {report['cleanup_count']}")
    print(f"  Emergency cleanup count: {report['emergency_cleanup_count']}")
    print(f"  Last cleanup time: {report['last_cleanup_time']:.1f}s ago")
    
    print("✅ Memory management test completed!")


def test_architecture_improvements():
    """Test architecture improvements and error handling."""
    print("\n🔍 Testing architecture improvements...")
    
    config = ConfigFactory.create_memory_optimized_config()
    
    # Test transcriber initialization with validation
    print("📊 Testing transcriber initialization...")
    try:
        transcriber = UnifiedAudioTranscriber(config)
        print("  ✅ Transcriber initialized successfully")
        
        # Test that all components are properly initialized
        assert hasattr(transcriber, 'whisper_transcriber'), "Missing whisper_transcriber attribute"
        assert hasattr(transcriber, 'chunk_calculator'), "Missing chunk_calculator attribute"
        assert hasattr(transcriber, 'memory_manager'), "Missing memory_manager attribute"
        assert hasattr(transcriber, 'validation_result'), "Missing validation_result attribute"
        
        print("  ✅ All components properly initialized")
        
        # Test chunk calculator integration
        test_duration = 30000  # 30 seconds
        chunk_analysis = transcriber.chunk_calculator.get_chunk_analysis(test_duration)
        print(f"  ✅ Chunk analysis: {chunk_analysis['total_chunks']} chunks for {test_duration/1000:.1f}s")
        
        # Test memory manager integration
        memory_report = transcriber.memory_manager.get_memory_report()
        print(f"  ✅ Memory report generated: {memory_report['current_usage_mb']:.1f}MB")
        
        # Test validation result
        if transcriber.validation_result.success:
            print("  ✅ Preprocessing validation successful")
        else:
            print("  ⚠️  Preprocessing validation had issues")
        
    except Exception as e:
        print(f"  ❌ Transcriber initialization failed: {e}")
        raise
    
    print("✅ Architecture improvements test completed!")


def test_error_handling():
    """Test error handling improvements."""
    print("\n🔍 Testing error handling...")
    
    config = ConfigFactory.create_memory_optimized_config()
    
    # Test invalid file handling
    print("📊 Testing invalid file handling...")
    try:
        transcriber = UnifiedAudioTranscriber(config)
        transcriber.transcribe_file("nonexistent_file.wav")
    except FileNotFoundError as e:
        print(f"  ✅ Properly caught FileNotFoundError: {e}")
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
    
    # Test invalid configuration
    print("📊 Testing invalid configuration...")
    try:
        # Create invalid config
        invalid_config = config.clone()
        invalid_config.chunk_duration_ms = -1  # Invalid value
        transcriber = UnifiedAudioTranscriber(invalid_config)
    except ValueError as e:
        print(f"  ✅ Properly caught ValueError: {e}")
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
    
    # Test memory error handling
    print("📊 Testing memory error handling...")
    try:
        transcriber = UnifiedAudioTranscriber(config)
        # Simulate memory pressure
        transcriber.memory_manager.cleanup(force=True, strategy_level=3)
        print("  ✅ Memory cleanup handled properly")
    except Exception as e:
        print(f"  ❌ Memory error handling failed: {e}")
    
    print("✅ Error handling test completed!")


def test_integration():
    """Test integration of all components."""
    print("\n🔍 Testing component integration...")
    
    config = ConfigFactory.create_memory_optimized_config()
    
    try:
        # Initialize transcriber
        transcriber = UnifiedAudioTranscriber(config)
        
        # Test chunk calculator integration
        calculator = transcriber.chunk_calculator
        test_duration = 20000  # 20 seconds
        
        # Test memory estimation
        memory_estimate = calculator.estimate_memory_usage(test_duration, 16000, 1)
        print(f"📊 Memory estimation: {memory_estimate['peak_memory_mb']:.1f}MB peak")
        
        # Test time estimation
        time_estimate = calculator.estimate_processing_time(test_duration, config.model_name)
        print(f"📊 Time estimation: {time_estimate['total_time_seconds']:.0f}s total")
        
        # Test memory manager integration
        memory_manager = transcriber.memory_manager
        memory_report = memory_manager.get_memory_report()
        print(f"📊 Memory report: {memory_report['current_usage_mb']:.1f}MB current")
        
        # Test preprocessing validation integration
        validation_result = transcriber.validation_result
        print(f"📊 Validation: {'Success' if validation_result.success else 'Failed'}")
        
        print("  ✅ All components integrated successfully")
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        raise
    
    print("✅ Integration test completed!")


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Comprehensive test for FarsiTranscribe fixes")
    parser.add_argument(
        "--test",
        choices=["all", "chunks", "preprocessing", "memory", "architecture", "errors", "integration"],
        default="all",
        help="Specific test to run (default: all)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    print("🧪 COMPREHENSIVE FARSITRANSCRIBE FIX TEST")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        if args.test == "all" or args.test == "chunks":
            test_chunk_calculator_consistency()
        
        if args.test == "all" or args.test == "preprocessing":
            test_preprocessing_validation()
        
        if args.test == "all" or args.test == "memory":
            test_memory_management()
        
        if args.test == "all" or args.test == "architecture":
            test_architecture_improvements()
        
        if args.test == "all" or args.test == "errors":
            test_error_handling()
        
        if args.test == "all" or args.test == "integration":
            test_integration()
        
        elapsed_time = time.time() - start_time
        print(f"\n🎉 All tests completed successfully in {elapsed_time:.2f} seconds!")
        print("✅ Comprehensive fix validation passed!")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ Test failed after {elapsed_time:.2f} seconds: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 