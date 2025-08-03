"""
Unified Memory and Performance Management for FarsiTranscribe
Consolidates functionality from multiple memory and performance management modules.
Provides comprehensive memory management with performance monitoring.
"""

import os
import gc
import time
import logging
import threading
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import psutil
import torch
import numpy as np


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    total_memory_mb: float
    available_memory_mb: float
    used_memory_mb: float
    memory_percent: float
    cpu_percent: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class MemoryThreshold:
    """Memory threshold configuration."""

    warning_threshold_mb: float = 1024  # 1GB
    critical_threshold_mb: float = 2048  # 2GB
    emergency_threshold_mb: float = 3072  # 3GB
    cleanup_threshold_mb: float = 512  # 512MB


@dataclass
class PerformanceMetrics:
    """Performance metrics data class."""

    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_chunks: int = 0
    processed_chunks: int = 0
    audio_duration: float = 0.0
    memory_usage_mb: List[float] = field(default_factory=list)
    cpu_usage_percent: List[float] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Get total processing duration."""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def chunks_per_second(self) -> float:
        """Get processing speed in chunks per second."""
        if self.duration == 0:
            return 0.0
        return self.processed_chunks / self.duration

    @property
    def audio_speedup(self) -> float:
        """Get speedup factor compared to real-time."""
        if self.audio_duration == 0:
            return 0.0
        return self.audio_duration / self.duration

    @property
    def avg_memory_usage(self) -> float:
        """Get average memory usage in MB."""
        if not self.memory_usage_mb:
            return 0.0
        return sum(self.memory_usage_mb) / len(self.memory_usage_mb)

    @property
    def avg_cpu_usage(self) -> float:
        """Get average CPU usage percentage."""
        if not self.cpu_usage_percent:
            return 0.0
        return sum(self.cpu_usage_percent) / len(self.cpu_usage_percent)


class MemoryMonitor:
    """Real-time memory monitoring with adaptive thresholds."""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.monitoring = False
        self.monitor_thread = None
        self.memory_history: List[MemoryStats] = []
        self.max_history_size = 100
        self.lock = threading.Lock()

        # Adaptive thresholds based on system memory
        self._setup_adaptive_thresholds()

    def _setup_adaptive_thresholds(self):
        """Setup adaptive memory thresholds based on system capabilities."""
        try:
            total_memory = psutil.virtual_memory().total / (1024 * 1024)  # MB

            # Adaptive thresholds based on total system memory
            if total_memory < 4096:  # Less than 4GB
                self.thresholds = MemoryThreshold(
                    warning_threshold_mb=total_memory * 0.3,
                    critical_threshold_mb=total_memory * 0.5,
                    emergency_threshold_mb=total_memory * 0.7,
                    cleanup_threshold_mb=total_memory * 0.2,
                )
            elif total_memory < 8192:  # Less than 8GB
                self.thresholds = MemoryThreshold(
                    warning_threshold_mb=1024,
                    critical_threshold_mb=2048,
                    emergency_threshold_mb=3072,
                    cleanup_threshold_mb=512,
                )
            else:  # 8GB or more
                self.thresholds = MemoryThreshold(
                    warning_threshold_mb=2048,
                    critical_threshold_mb=4096,
                    emergency_threshold_mb=6144,
                    cleanup_threshold_mb=1024,
                )

            self.logger.info(
                f"Memory thresholds configured for {total_memory:.0f}MB system:"
            )
            self.logger.info(
                f"  Warning: {self.thresholds.warning_threshold_mb:.0f}MB"
            )
            self.logger.info(
                f"  Critical: {self.thresholds.critical_threshold_mb:.0f}MB"
            )
            self.logger.info(
                f"  Emergency: {self.thresholds.emergency_threshold_mb:.0f}MB"
            )
            self.logger.info(
                f"  Cleanup: {self.thresholds.cleanup_threshold_mb:.0f}MB"
            )

        except Exception as e:
            self.logger.warning(f"Failed to setup adaptive thresholds: {e}")
            # Fallback to default thresholds
            self.thresholds = MemoryThreshold()

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)

            stats = MemoryStats(
                total_memory_mb=memory.total / (1024 * 1024),
                available_memory_mb=memory.available / (1024 * 1024),
                used_memory_mb=memory.used / (1024 * 1024),
                memory_percent=memory.percent,
                cpu_percent=cpu_percent,
            )

            # Store in history
            with self.lock:
                self.memory_history.append(stats)
                if len(self.memory_history) > self.max_history_size:
                    self.memory_history.pop(0)

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            return MemoryStats(0, 0, 0, 0, 0)

    def start_monitoring(self, interval: float = 5.0):
        """Start continuous memory monitoring."""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,), daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("Memory monitoring started")

    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        self.logger.info("Memory monitoring stopped")

    def _monitor_loop(self, interval: float):
        """Memory monitoring loop."""
        while self.monitoring:
            try:
                stats = self.get_memory_stats()
                self._check_memory_status(stats)
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")
                time.sleep(interval)

    def _check_memory_status(self, stats: MemoryStats):
        """Check memory status and log warnings if needed."""
        if stats.used_memory_mb > self.thresholds.emergency_threshold_mb:
            self.logger.critical(
                f"EMERGENCY: Memory usage at {stats.used_memory_mb:.0f}MB"
            )
        elif stats.used_memory_mb > self.thresholds.critical_threshold_mb:
            self.logger.error(
                f"CRITICAL: Memory usage at {stats.used_memory_mb:.0f}MB"
            )
        elif stats.used_memory_mb > self.thresholds.warning_threshold_mb:
            self.logger.warning(
                f"WARNING: Memory usage at {stats.used_memory_mb:.0f}MB"
            )

    def get_memory_trend(self) -> Dict[str, float]:
        """Get memory usage trend over time."""
        with self.lock:
            if len(self.memory_history) < 2:
                return {"trend": 0.0, "avg_usage": 0.0}

            recent = self.memory_history[-10:]  # Last 10 samples
            avg_usage = sum(s.used_memory_mb for s in recent) / len(recent)

            if len(recent) >= 2:
                trend = (
                    recent[-1].used_memory_mb - recent[0].used_memory_mb
                ) / len(recent)
            else:
                trend = 0.0

            return {
                "trend": trend,
                "avg_usage": avg_usage,
                "samples": len(recent),
            }


class UnifiedMemoryManager:
    """
    Unified memory manager with performance monitoring capabilities.
    Consolidates functionality from multiple memory management modules.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.monitor = MemoryMonitor(config)
        self.last_cleanup_time = 0
        self.cleanup_count = 0
        self.emergency_cleanup_count = 0
        self.lock = threading.Lock()

        # Performance monitoring
        self.performance_metrics = PerformanceMetrics()
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()

        # Cleanup strategies
        self.cleanup_strategies = [
            self._light_cleanup,
            self._medium_cleanup,
            self._heavy_cleanup,
            self._emergency_cleanup,
        ]

        # Start monitoring
        self.monitor.start_monitoring()

    def __del__(self):
        """Cleanup on destruction."""
        self.monitor.stop_monitoring()
        self.stop_performance_monitoring()

    def check_memory_usage(self) -> bool:
        """Check if memory usage is within acceptable limits."""
        stats = self.monitor.get_memory_stats()
        return (
            stats.used_memory_mb < self.monitor.thresholds.warning_threshold_mb
        )

    def should_cleanup(self) -> bool:
        """Determine if cleanup is needed."""
        stats = self.monitor.get_memory_stats()
        time_since_cleanup = time.time() - self.last_cleanup_time

        # Check various conditions for cleanup
        conditions = [
            stats.used_memory_mb
            > self.monitor.thresholds.cleanup_threshold_mb,
            time_since_cleanup > self.config.cleanup_interval_seconds,
            self.cleanup_count == 0,  # First cleanup
            self._is_memory_trend_increasing(),
        ]

        return any(conditions)

    def _is_memory_trend_increasing(self) -> bool:
        """Check if memory usage is trending upward."""
        trend = self.monitor.get_memory_trend()
        return trend["trend"] > 50  # 50MB per sample increasing trend

    def cleanup(self, force: bool = False, strategy_level: int = 0):
        """
        Perform memory cleanup with adaptive strategy selection.

        Args:
            force: Force cleanup regardless of conditions
            strategy_level: Specific cleanup strategy to use (0-3)
        """
        with self.lock:
            stats = self.monitor.get_memory_stats()

            if not force and not self.should_cleanup():
                return

            # Select cleanup strategy based on memory pressure
            if strategy_level == 0:
                if (
                    stats.used_memory_mb
                    > self.monitor.thresholds.emergency_threshold_mb
                ):
                    strategy_level = 3  # Emergency cleanup
                elif (
                    stats.used_memory_mb
                    > self.monitor.thresholds.critical_threshold_mb
                ):
                    strategy_level = 2  # Heavy cleanup
                elif (
                    stats.used_memory_mb
                    > self.monitor.thresholds.warning_threshold_mb
                ):
                    strategy_level = 1  # Medium cleanup
                else:
                    strategy_level = 0  # Light cleanup

            # Execute cleanup strategy
            try:
                if 0 <= strategy_level < len(self.cleanup_strategies):
                    cleanup_func = self.cleanup_strategies[strategy_level]
                    cleanup_func()

                    if strategy_level == 3:
                        self.emergency_cleanup_count += 1
                    else:
                        self.cleanup_count += 1

                    self.last_cleanup_time = time.time()

                    # Log cleanup results
                    new_stats = self.monitor.get_memory_stats()
                    memory_freed = (
                        stats.used_memory_mb - new_stats.used_memory_mb
                    )

                    self.logger.info(
                        f"Memory cleanup completed (strategy {strategy_level}): "
                        f"Freed {memory_freed:.1f}MB, "
                        f"Current usage: {new_stats.used_memory_mb:.1f}MB"
                    )

            except Exception as e:
                self.logger.error(f"Memory cleanup failed: {e}")

    def _light_cleanup(self):
        """Light cleanup: Basic garbage collection."""
        gc.collect()

        # Clear Python cache
        if hasattr(gc, "garbage"):
            gc.garbage.clear()

    def _medium_cleanup(self):
        """Medium cleanup: Enhanced garbage collection and cache clearing."""
        self._light_cleanup()

        # Clear NumPy cache
        try:
            np.core.multiarray._get_promotion_state.cache_clear()
        except:
            pass

        # Clear torch cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _heavy_cleanup(self):
        """Heavy cleanup: Aggressive memory management."""
        self._medium_cleanup()

        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()

        # Clear more caches
        try:
            import sys

            # Clear some common caches
            for module_name in ["numpy", "scipy", "librosa"]:
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name, None)
                        if hasattr(attr, "cache_clear"):
                            try:
                                attr.cache_clear()
                            except:
                                pass
        except:
            pass

    def _emergency_cleanup(self):
        """Emergency cleanup: Maximum memory recovery."""
        self._heavy_cleanup()

        # Force memory compaction
        try:
            import ctypes

            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except:
            pass

        # Additional aggressive cleanup
        gc.collect()
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # Performance monitoring methods
    def start_performance_monitoring(
        self, total_chunks: int, audio_duration: float
    ):
        """Start performance monitoring."""
        self.performance_metrics = PerformanceMetrics()
        self.performance_metrics.total_chunks = total_chunks
        self.performance_metrics.audio_duration = audio_duration
        self.stop_monitoring.clear()

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitor_resources
        )
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

    def stop_performance_monitoring(self):
        """Stop performance monitoring."""
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)

        self.performance_metrics.end_time = time.time()

    def update_progress(self, processed_chunks: int):
        """Update processing progress."""
        self.performance_metrics.processed_chunks = processed_chunks

    def _monitor_resources(self):
        """Monitor system resources in background thread."""
        while not self.stop_monitoring.wait(2.0):  # Sample every 2 seconds
            try:
                # Memory usage
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.performance_metrics.memory_usage_mb.append(memory_mb)

                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.performance_metrics.cpu_usage_percent.append(cpu_percent)

            except Exception:
                # Ignore monitoring errors
                pass

    @contextmanager
    def memory_context(self, max_memory_mb: Optional[float] = None):
        """
        Context manager for memory-intensive operations.

        Args:
            max_memory_mb: Maximum memory usage allowed during operation
        """
        initial_stats = self.monitor.get_memory_stats()

        try:
            yield self
        finally:
            final_stats = self.monitor.get_memory_stats()
            memory_increase = (
                final_stats.used_memory_mb - initial_stats.used_memory_mb
            )

            if memory_increase > 100:  # More than 100MB increase
                self.logger.info(
                    f"Memory context: {memory_increase:.1f}MB increase, triggering cleanup"
                )
                self.cleanup(force=True)

    def get_memory_report(self) -> Dict:
        """Get comprehensive memory usage report."""
        stats = self.monitor.get_memory_stats()
        trend = self.monitor.get_memory_trend()

        return {
            "current_usage_mb": stats.used_memory_mb,
            "available_memory_mb": stats.available_memory_mb,
            "total_memory_mb": stats.total_memory_mb,
            "memory_percent": stats.memory_percent,
            "cpu_percent": stats.cpu_percent,
            "memory_trend": trend,
            "cleanup_count": self.cleanup_count,
            "emergency_cleanup_count": self.emergency_cleanup_count,
            "last_cleanup_time": self.last_cleanup_time,
            "thresholds": {
                "warning_mb": self.monitor.thresholds.warning_threshold_mb,
                "critical_mb": self.monitor.thresholds.critical_threshold_mb,
                "emergency_mb": self.monitor.thresholds.emergency_threshold_mb,
                "cleanup_mb": self.monitor.thresholds.cleanup_threshold_mb,
            },
        }

    def get_performance_summary(self) -> Dict[str, any]:
        """Get performance summary."""
        return {
            "duration_seconds": self.performance_metrics.duration,
            "audio_duration_seconds": self.performance_metrics.audio_duration,
            "speedup_factor": self.performance_metrics.audio_speedup,
            "chunks_per_second": self.performance_metrics.chunks_per_second,
            "total_chunks": self.performance_metrics.total_chunks,
            "processed_chunks": self.performance_metrics.processed_chunks,
            "avg_memory_mb": self.performance_metrics.avg_memory_usage,
            "avg_cpu_percent": self.performance_metrics.avg_cpu_usage,
            "efficiency_score": self._calculate_efficiency_score(),
        }

    def _calculate_efficiency_score(self) -> float:
        """Calculate overall efficiency score (0-100)."""
        if (
            self.performance_metrics.duration == 0
            or self.performance_metrics.audio_duration == 0
        ):
            return 0.0

        # Factors to consider:
        # 1. Speedup factor (higher is better)
        # 2. Memory efficiency (lower is better)
        # 3. CPU utilization (moderate is better)

        speedup_score = min(
            100, self.performance_metrics.audio_speedup * 10
        )  # Cap at 100

        memory_score = max(
            0, 100 - (self.performance_metrics.avg_memory_usage / 100)
        )  # Penalize high memory

        cpu_score = 100 - abs(
            self.performance_metrics.avg_cpu_usage - 80
        )  # Optimal around 80%

        # Weighted average
        return speedup_score * 0.5 + memory_score * 0.3 + cpu_score * 0.2

    def print_performance_summary(self):
        """Print performance summary to console."""
        summary = self.get_performance_summary()

        print("\n" + "=" * 60)
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"⏱️  Processing Time: {summary['duration_seconds']:.1f}s")
        print(f"🎵 Audio Duration: {summary['audio_duration_seconds']:.1f}s")
        print(f"⚡ Speedup Factor: {summary['speedup_factor']:.2f}x")
        print(
            f"🚀 Processing Speed: {summary['chunks_per_second']:.2f} chunks/s"
        )
        print(
            f"📦 Chunks Processed: {summary['processed_chunks']}/{summary['total_chunks']}"
        )
        print(f"💾 Avg Memory Usage: {summary['avg_memory_mb']:.1f} MB")
        print(f"🖥️  Avg CPU Usage: {summary['avg_cpu_percent']:.1f}%")
        print(f"🎯 Efficiency Score: {summary['efficiency_score']:.1f}/100")
        print("=" * 60)

    def optimize_for_operation(
        self, operation_type: str, estimated_memory_mb: float
    ):
        """
        Optimize memory for a specific operation.

        Args:
            operation_type: Type of operation ("transcription", "preprocessing", etc.)
            estimated_memory_mb: Estimated memory usage for the operation
        """
        stats = self.monitor.get_memory_stats()
        available_memory = stats.available_memory_mb

        if estimated_memory_mb > available_memory * 0.8:
            self.logger.warning(
                f"Operation {operation_type} may require {estimated_memory_mb:.1f}MB, "
                f"but only {available_memory:.1f}MB available. Triggering preemptive cleanup."
            )
            self.cleanup(force=True, strategy_level=1)

        # Adjust thresholds temporarily for this operation
        original_cleanup = self.monitor.thresholds.cleanup_threshold_mb
        self.monitor.thresholds.cleanup_threshold_mb = min(
            original_cleanup, available_memory * 0.3
        )

        return original_cleanup  # Return original value for restoration

    def restore_thresholds(self, original_cleanup_threshold: float):
        """Restore original memory thresholds."""
        self.monitor.thresholds.cleanup_threshold_mb = (
            original_cleanup_threshold
        )


@contextmanager
def performance_monitor(enable_monitoring: bool = True):
    """Context manager for performance monitoring."""

    # This is a simplified version for backward compatibility
    # In practice, you would use the UnifiedMemoryManager
    class DummyMonitor:
        def __init__(self):
            pass

        def start_monitoring(self, *args):
            pass

        def stop_monitoring(self):
            pass

        def update_progress(self, *args):
            pass

        def print_summary(self):
            pass

    monitor = DummyMonitor()
    try:
        yield monitor
    finally:
        monitor.stop_monitoring()


def create_unified_memory_manager(config) -> UnifiedMemoryManager:
    """
    Factory function to create a unified memory manager.

    Args:
        config: TranscriptionConfig object

    Returns:
        UnifiedMemoryManager instance
    """
    return UnifiedMemoryManager(config)
