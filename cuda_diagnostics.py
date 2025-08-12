#!/usr/bin/env python3
"""
Comprehensive CUDA diagnostics to identify GPU compatibility issues.
"""

import sys

import torch
import torch.nn as nn
import numpy as np

def test_cuda_basic():
    """Test basic CUDA functionality."""
    print("🔍 Basic CUDA Tests")
    print("=" * 40)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                print(f"Device {i}: {props.name}")
                print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
                print(f"  Compute Capability: {props.major}.{props.minor}")
                print(f"  Multi-processor count: {props.multi_processor_count}")
            except Exception as e:
                print(f"  Error getting device {i} properties: {e}")
    else:
        print("❌ CUDA not available")

def test_cuda_operations():
    """Test CUDA operations step by step."""
    print("\n🧪 CUDA Operation Tests")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping operation tests")
        return
    
    try:
        # Test 1: Basic tensor creation
        print("1. Testing tensor creation...")
        test_tensor = torch.randn(2, 2)
        print("   ✅ CPU tensor creation successful")
        
        # Test 2: Move to CUDA
        print("2. Testing tensor move to CUDA...")
        cuda_tensor = test_tensor.cuda()
        print("   ✅ Tensor moved to CUDA successfully")
        
        # Test 3: Basic operations
        print("3. Testing basic operations...")
        result = torch.matmul(cuda_tensor, cuda_tensor)
        print("   ✅ Matrix multiplication successful")
        
        # Test 4: Synchronization
        print("4. Testing CUDA synchronization...")
        torch.cuda.synchronize()
        print("   ✅ CUDA synchronization successful")
        
        # Test 5: Memory operations
        print("5. Testing memory operations...")
        torch.cuda.empty_cache()
        print("   ✅ CUDA memory cache cleared")
        
        print("\n✅ All CUDA operations successful!")
        
    except Exception as e:
        print(f"❌ CUDA operation failed: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

def test_cuda_memory():
    """Test CUDA memory management."""
    print("\n💾 CUDA Memory Tests")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping memory tests")
        return
    
    try:
        # Get initial memory info
        initial_memory = torch.cuda.memory_allocated()
        initial_cached = torch.cuda.memory_reserved()
        
        print(f"Initial allocated: {initial_memory / 1e6:.1f} MB")
        print(f"Initial cached: {initial_cached / 1e6:.1f} MB")
        
        # Test memory allocation
        print("Testing memory allocation...")
        test_tensors = []
        for i in range(5):
            tensor = torch.randn(1000, 1000, device='cuda')
            test_tensors.append(tensor)
            current_memory = torch.cuda.memory_allocated()
            print(f"  After tensor {i+1}: {current_memory / 1e6:.1f} MB")
        
        # Test memory cleanup
        print("Testing memory cleanup...")
        del test_tensors
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated()
        final_cached = torch.cuda.memory_reserved()
        
        print(f"Final allocated: {final_memory / 1e6:.1f} MB")
        print(f"Final cached: {final_cached / 1e6:.1f} MB")
        
        print("✅ CUDA memory management successful!")
        
    except Exception as e:
        print(f"❌ CUDA memory test failed: {e}")

def test_cuda_compatibility():
    """Test CUDA compatibility with different data types."""
    print("\n🔧 CUDA Compatibility Tests")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping compatibility tests")
        return
    
    try:
        # Test different data types
        dtypes = [torch.float32, torch.float16, torch.int32, torch.int64]
        
        for dtype in dtypes:
            print(f"Testing {dtype}...")
            try:
                tensor = torch.randn(100, 100, dtype=dtype, device='cuda')
                result = torch.matmul(tensor, tensor)
                print(f"  ✅ {dtype} operations successful")
            except Exception as e:
                print(f"  ❌ {dtype} operations failed: {e}")
        
        # Test model operations
        print("Testing model operations...")
        try:
            model = nn.Linear(100, 100).cuda()
            input_tensor = torch.randn(10, 100, device='cuda')
            output = model(input_tensor)
            print("  ✅ Model operations successful")
        except Exception as e:
            print(f"  ❌ Model operations failed: {e}")
            
    except Exception as e:
        print(f"❌ CUDA compatibility test failed: {e}")

def test_device_specific():
    """Test specific device operations."""
    print("\n🎯 Device-Specific Tests")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping device tests")
        return
    
    try:
        # Test specific device
        device = 'cuda:0'
        print(f"Testing device: {device}")
        
        # Test tensor operations on specific device
        tensor = torch.randn(2, 2, device=device)
        print(f"  ✅ Tensor created on {device}")
        
        # Test operations
        result = torch.matmul(tensor, tensor)
        print(f"  ✅ Operations on {device} successful")
        
        # Test synchronization
        torch.cuda.synchronize(device)
        print(f"  ✅ Synchronization on {device} successful")
        
        print(f"✅ Device {device} is working properly!")
        
    except Exception as e:
        print(f"❌ Device test failed: {e}")

def main():
    """Run all CUDA diagnostics."""
    print("🚀 CUDA Diagnostics for FarsiTranscribe")
    print("=" * 50)
    
    test_cuda_basic()
    test_cuda_operations()
    test_cuda_memory()
    test_cuda_compatibility()
    test_device_specific()
    
    print("\n" + "=" * 50)
    print("🏁 CUDA Diagnostics Complete")
    
    # Summary
    if torch.cuda.is_available():
        print("✅ CUDA is available")
        print("📊 Check the test results above for specific issues")
    else:
        print("❌ CUDA is not available")

if __name__ == "__main__":
    main()
