#!/usr/bin/env python3
"""
Test script to verify device detection and fallback mechanisms.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from src.core.transcriber import DeviceManager
from src.core.config import TranscriptionConfig

def test_device_detection():
    """Test the device detection system."""
    print("🔍 Testing Device Detection System")
    print("=" * 40)
    
    # Create config
    config = TranscriptionConfig()
    print(f"Initial device: {config.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        try:
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
            print(f"CUDA device 0 memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        except Exception as e:
            print(f"CUDA device info failed: {e}")
    
    # Create device manager
    device_manager = DeviceManager(config)
    print(f"Selected device: {device_manager.get_device()}")
    print(f"Device fallback history: {device_manager.device_fallback_history}")
    
    # Test device functionality
    print(f"\n🧪 Testing device functionality...")
    is_working = device_manager.is_cuda_available()
    print(f"CUDA working: {is_working}")
    
    # Test CPU fallback
    print(f"\n🔄 Testing CPU fallback...")
    device_manager.force_cpu_fallback()
    print(f"Device after fallback: {device_manager.get_device()}")
    print(f"Config device: {config.device}")
    print(f"Config batch size: {config.batch_size}")
    
    print("\n✅ Device detection test completed!")

if __name__ == "__main__":
    test_device_detection()
