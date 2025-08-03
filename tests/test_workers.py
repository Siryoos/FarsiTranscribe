#!/usr/bin/env python3
"""
Test script to verify worker configuration.
"""

import sys
import os

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_worker_config():
    """Test that all configurations use appropriate number of workers."""
    print("🔧 Testing Worker Configuration")
    print("=" * 50)

    try:
        from src.core.config import ConfigFactory

        # Test Persian optimized config
        persian_config = ConfigFactory.create_persian_optimized_config()
        print(
            f"✅ Persian optimized config: {persian_config.num_workers} workers"
        )
        assert persian_config.num_workers > 0

        # Test fast config
        fast_config = ConfigFactory.create_fast_config()
        print(f"✅ Fast config: {fast_config.num_workers} workers")
        assert fast_config.num_workers > 0

        # Test high quality config
        quality_config = ConfigFactory.create_high_quality_config()
        print(f"✅ High quality config: {quality_config.num_workers} workers")
        assert quality_config.num_workers > 0

        # Test optimized config
        optimized_config = ConfigFactory.create_optimized_config()
        print(f"✅ Optimized config: {optimized_config.num_workers} workers")
        assert optimized_config.num_workers > 0

        # Test default config
        from src.core.config import TranscriptionConfig

        default_config = TranscriptionConfig()
        print(f"✅ Default config: {default_config.num_workers} workers")
        assert default_config.num_workers > 0

        # Verify all have positive worker counts
        configs = [
            persian_config,
            fast_config,
            quality_config,
            optimized_config,
            default_config,
        ]
        all_valid = all(config.num_workers > 0 for config in configs)

        if all_valid:
            print("\n🎉 All configurations have valid worker counts!")
            assert True
        else:
            print("\n❌ Some configurations have invalid worker counts!")
            assert False, "Some configurations have invalid worker counts"

    except ImportError as e:
        print(f"❌ Import error: {e}")
        assert False, f"Import error: {e}"
    except Exception as e:
        print(f"❌ Error: {e}")
        assert False, f"Error: {e}"


if __name__ == "__main__":
    success = test_worker_config()
    sys.exit(0 if success else 1)
