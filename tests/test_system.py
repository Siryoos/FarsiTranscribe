#!/usr/bin/env python3
"""
Comprehensive system test for FarsiTranscribe.
Tests all major components and Unicode support.
"""

import sys
import os

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_imports():
    """Test all major imports."""
    print("🔍 Testing imports...")

    try:
        from src.core.config import TranscriptionConfig, ConfigFactory

        print("✅ Configuration module imported successfully")
    except ImportError as e:
        print(f"❌ Configuration import failed: {e}")
        return False

    try:
        from src.utils.terminal_display import terminal_display

        print("✅ Terminal display module imported successfully")
    except ImportError as e:
        print(f"❌ Terminal display import failed: {e}")
        return False

    try:
        from src.utils.sentence_extractor import SentenceExtractor

        print("✅ Sentence extractor module imported successfully")
    except ImportError as e:
        print(f"❌ Sentence extractor import failed: {e}")
        return False

    try:
        from src.utils.repetition_detector import RepetitionDetector

        print("✅ Repetition detector module imported successfully")
    except ImportError as e:
        print(f"❌ Repetition detector import failed: {e}")
        return False

    return True


def test_unicode_support():
    """Test Unicode support for Persian text."""
    print("\n🔍 Testing Unicode support...")

    try:
        from src.utils.terminal_display import terminal_display

        # Test Persian text
        persian_text = "سلام دنیا، این یک تست است"

        print("📝 Testing Persian text display:")
        terminal_display.print_persian_preview(persian_text, 1)

        # Test Unicode support check
        if terminal_display.check_unicode_support():
            print("✅ Unicode support is working properly")
            return True
        else:
            print("⚠️ Unicode support may have issues")
            return False

    except Exception as e:
        print(f"❌ Unicode test failed: {e}")
        return False


def test_configuration():
    """Test configuration system."""
    print("\n🔍 Testing configuration system...")

    try:
        from src.core.config import TranscriptionConfig, ConfigFactory

        # Test basic configuration
        config = TranscriptionConfig()
        print(f"✅ Basic configuration created: {config.model_name} model")

        # Test factory methods
        fast_config = ConfigFactory.create_fast_config()
        print(f"✅ Fast configuration created: {fast_config.model_name} model")

        quality_config = ConfigFactory.create_high_quality_config()
        print(
            f"✅ Quality configuration created: {quality_config.model_name} model"
        )

        optimized_config = ConfigFactory.create_optimized_config()
        print(
            f"✅ Optimized configuration created: {optimized_config.model_name} model"
        )

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_utilities():
    """Test utility functions."""
    print("\n🔍 Testing utility functions...")

    try:
        from src.utils.sentence_extractor import SentenceExtractor
        from src.utils.repetition_detector import RepetitionDetector
        from src.core.config import TranscriptionConfig

        # Test sentence extraction
        test_text = "این جمله اول است. این جمله دوم است. و این جمله سوم است."
        sentences = SentenceExtractor.extract_sentences(test_text, 3)
        print(f"✅ Sentence extraction: {len(sentences)} sentences extracted")

        # Test repetition detection
        repetitive_text = "سلام سلام سلام دنیا دنیا دنیا"
        cleaned_text = RepetitionDetector.clean_repetitive_text(
            repetitive_text, TranscriptionConfig()
        )
        print(
            f"✅ Repetition detection: Text cleaned from {len(repetitive_text)} to {len(cleaned_text)} characters"
        )

        return True

    except Exception as e:
        print(f"❌ Utility test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🎙️ FarsiTranscribe System Test")
    print("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("Unicode Support", test_unicode_support),
        ("Configuration", test_configuration),
        ("Utilities", test_utilities),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
