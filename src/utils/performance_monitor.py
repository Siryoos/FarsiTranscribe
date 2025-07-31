"""
Performance monitoring utilities for transcription system.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager


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


class PerformanceMonitor:
    """Performance monitoring utility."""
    
    def __init__(self, enable_monitoring: bool = True):
        self.enable_monitoring = enable_monitoring
        self.metrics = PerformanceMetrics()
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
    def start_monitoring(self, total_chunks: int, audio_duration: float):
        """Start performance monitoring."""
        if not self.enable_monitoring:
            return
            
        self.metrics = PerformanceMetrics()
        self.metrics.total_chunks = total_chunks
        self.metrics.audio_duration = audio_duration
        self.stop_monitoring.clear()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_resources)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self.enable_monitoring:
            return
            
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        
        self.metrics.end_time = time.time()
    
    def update_progress(self, processed_chunks: int):
        """Update processing progress."""
        if self.enable_monitoring:
            self.metrics.processed_chunks = processed_chunks
    
    def _monitor_resources(self):
        """Monitor system resources in background thread."""
        while not self.stop_monitoring.wait(2.0):  # Sample every 2 seconds
            try:
                # Memory usage
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.metrics.memory_usage_mb.append(memory_mb)
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.metrics.cpu_usage_percent.append(cpu_percent)
                
            except Exception:
                # Ignore monitoring errors
                pass
    
    def get_summary(self) -> Dict[str, any]:
        """Get performance summary."""
        if not self.enable_monitoring:
            return {}
        
        return {
            'duration_seconds': self.metrics.duration,
            'audio_duration_seconds': self.metrics.audio_duration,
            'speedup_factor': self.metrics.audio_speedup,
            'chunks_per_second': self.metrics.chunks_per_second,
            'total_chunks': self.metrics.total_chunks,
            'processed_chunks': self.metrics.processed_chunks,
            'avg_memory_mb': self.metrics.avg_memory_usage,
            'avg_cpu_percent': self.metrics.avg_cpu_usage,
            'efficiency_score': self._calculate_efficiency_score()
        }
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate overall efficiency score (0-100)."""
        if self.metrics.duration == 0 or self.metrics.audio_duration == 0:
            return 0.0
        
        # Factors to consider:
        # 1. Speedup factor (higher is better)
        # 2. Memory efficiency (lower is better)
        # 3. CPU utilization (moderate is better)
        
        speedup_score = min(100, self.metrics.audio_speedup * 10)  # Cap at 100
        
        memory_score = max(0, 100 - (self.metrics.avg_memory_usage / 100))  # Penalize high memory
        
        cpu_score = 100 - abs(self.metrics.avg_cpu_usage - 80)  # Optimal around 80%
        
        # Weighted average
        return (speedup_score * 0.5 + memory_score * 0.3 + cpu_score * 0.2)
    
    def print_summary(self):
        """Print performance summary to console."""
        if not self.enable_monitoring:
            return
        
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"⏱️  Processing Time: {summary['duration_seconds']:.1f}s")
        print(f"🎵 Audio Duration: {summary['audio_duration_seconds']:.1f}s")
        print(f"⚡ Speedup Factor: {summary['speedup_factor']:.2f}x")
        print(f"🚀 Processing Speed: {summary['chunks_per_second']:.2f} chunks/s")
        print(f"📦 Chunks Processed: {summary['processed_chunks']}/{summary['total_chunks']}")
        print(f"💾 Avg Memory Usage: {summary['avg_memory_mb']:.1f} MB")
        print(f"🖥️  Avg CPU Usage: {summary['avg_cpu_percent']:.1f}%")
        print(f"🎯 Efficiency Score: {summary['efficiency_score']:.1f}/100")
        print("=" * 60)


@contextmanager
def performance_monitor(enable_monitoring: bool = True):
    """Context manager for performance monitoring."""
    monitor = PerformanceMonitor(enable_monitoring)
    try:
        yield monitor
    finally:
        monitor.stop_monitoring()
        if enable_monitoring:
            monitor.print_summary() 