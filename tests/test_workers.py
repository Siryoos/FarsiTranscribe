#!/usr/bin/env python3
"""
Test script to verify worker configuration.
"""

import sys
import os

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_worker_config():
    """Test that all configurations use 4 workers."""
    print("🔧 Testing Worker Configuration")
    print("=" * 50)

    try:
        from src.core.config import ConfigFactory

        # Test Persian optimized config
        persian_config = ConfigFactory.create_persian_optimized_config()
        print(
            f"✅ Persian optimized config: {persian_config.num_workers} workers"
        )

        # Test fast config
        fast_config = ConfigFactory.create_fast_config()
        print(f"✅ Fast config: {fast_config.num_workers} workers")

        # Test high quality config
        quality_config = ConfigFactory.create_high_quality_config()
        print(f"✅ High quality config: {quality_config.num_workers} workers")

        # Test optimized config
        optimized_config = ConfigFactory.create_optimized_config()
        print(f"✅ Optimized config: {optimized_config.num_workers} workers")

        # Test default config
        from src.core.config import TranscriptionConfig

        default_config = TranscriptionConfig()
        print(f"✅ Default config: {default_config.num_workers} workers")

        # Verify all are 2
        configs = [
            persian_config,
            fast_config,
            quality_config,
            optimized_config,
            default_config,
        ]
        all_two = all(config.num_workers == 2 for config in configs)

        if all_two:
            print("\n🎉 All configurations are using 2 workers!")
        else:
            print("\n❌ Some configurations are not using 2 workers!")

        return all_two

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = test_worker_config()
    sys.exit(0 if success else 1)
