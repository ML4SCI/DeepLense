"""
Performance monitoring framework for RIPPLe data access.

This module provides comprehensive performance monitoring with metrics collection,
memory tracking, timing analysis, and regression detection.
"""

import logging
import time
import threading
import psutil
import gc
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    name: str
    metric_type: MetricType
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    description: Optional[str] = None


@dataclass
class PerformanceAlert:
    """Performance alert notification."""
    level: AlertLevel
    metric_name: str
    message: str
    value: float
    threshold: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""
    metric_name: str
    threshold_value: float
    alert_level: AlertLevel
    comparison: str = "greater"  # "greater", "less", "equal"
    description: Optional[str] = None


class MemoryTracker:
    """Memory usage tracking utilities."""
    
    def __init__(self, sample_interval: float = 1.0):
        self.sample_interval = sample_interval
        self.memory_samples: deque = deque(maxlen=1000)
        self.process = psutil.Process()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Start background monitoring
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_memory(self):
        """Background thread for memory monitoring."""
        while self.monitoring:
            try:
                # Get memory info
                memory_info = self.process.memory_info()
                
                sample = {
                    'timestamp': time.time(),
                    'rss': memory_info.rss,  # Resident Set Size
                    'vms': memory_info.vms,  # Virtual Memory Size
                    'percent': self.process.memory_percent(),
                    'available': psutil.virtual_memory().available
                }
                
                self.memory_samples.append(sample)
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")
                time.sleep(self.sample_interval)
    
    def get_current_memory(self) -> Dict[str, float]:
        """Get current memory usage."""
        try:
            memory_info = self.process.memory_info()
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': self.process.memory_percent(),
                'available_mb': psutil.virtual_memory().available / 1024 / 1024
            }
        except Exception as e:
            self.logger.error(f"Failed to get current memory: {e}")
            return {}
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        if not self.memory_samples:
            return 0.0
        return max(sample['rss'] for sample in self.memory_samples) / 1024 / 1024
    
    def get_memory_trend(self, window_size: int = 10) -> str:
        """Get memory usage trend (increasing/decreasing/stable)."""
        if len(self.memory_samples) < window_size:
            return "insufficient_data"
        
        recent_samples = list(self.memory_samples)[-window_size:]
        first_half = recent_samples[:window_size//2]
        second_half = recent_samples[window_size//2:]
        
        first_avg = sum(s['rss'] for s in first_half) / len(first_half)
        second_avg = sum(s['rss'] for s in second_half) / len(second_half)
        
        diff_percent = (second_avg - first_avg) / first_avg * 100
        
        if diff_percent > 5:
            return "increasing"
        elif diff_percent < -5:
            return "decreasing"
        else:
            return "stable"
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        if not self.memory_samples:
            return {}
        
        rss_values = [s['rss'] for s in self.memory_samples]
        percent_values = [s['percent'] for s in self.memory_samples]
        
        return {
            'current_rss_mb': rss_values[-1] / 1024 / 1024,
            'peak_rss_mb': max(rss_values) / 1024 / 1024,
            'min_rss_mb': min(rss_values) / 1024 / 1024,
            'avg_rss_mb': sum(rss_values) / len(rss_values) / 1024 / 1024,
            'current_percent': percent_values[-1],
            'peak_percent': max(percent_values),
            'trend': self.get_memory_trend(),
            'sample_count': len(self.memory_samples)
        }
    
    def force_gc(self) -> Dict[str, int]:
        """Force garbage collection and return statistics."""
        try:
            collected = gc.collect()
            stats = {
                'collected_objects': collected,
                'garbage_count': len(gc.garbage),
                'generation_counts': gc.get_count()
            }
            return stats
        except Exception as e:
            self.logger.error(f"Failed to force GC: {e}")
            return {}
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)


class TimingProfiler:
    """Timing profiler for performance analysis."""
    
    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self.timing_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self.active_timers: Dict[str, float] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def start_timer(self, operation_name: str, context: Optional[Dict[str, Any]] = None):
        """Start timing an operation."""
        timer_key = f"{operation_name}_{threading.current_thread().ident}"
        self.active_timers[timer_key] = time.time()
        
        if context:
            self.active_timers[f"{timer_key}_context"] = context
    
    def end_timer(self, operation_name: str) -> float:
        """End timing an operation and return duration."""
        timer_key = f"{operation_name}_{threading.current_thread().ident}"
        
        if timer_key not in self.active_timers:
            self.logger.warning(f"No active timer for operation: {operation_name}")
            return 0.0
        
        start_time = self.active_timers.pop(timer_key)
        duration = time.time() - start_time
        
        # Store timing data
        self.timing_data[operation_name].append({
            'duration': duration,
            'timestamp': time.time(),
            'context': self.active_timers.pop(f"{timer_key}_context", None)
        })
        
        return duration
    
    def get_timing_stats(self, operation_name: str) -> Dict[str, float]:
        """Get timing statistics for an operation."""
        if operation_name not in self.timing_data:
            return {}
        
        durations = [sample['duration'] for sample in self.timing_data[operation_name]]
        
        if not durations:
            return {}
        
        return {
            'count': len(durations),
            'mean': sum(durations) / len(durations),
            'min': min(durations),
            'max': max(durations),
            'median': sorted(durations)[len(durations)//2],
            'p95': sorted(durations)[int(len(durations)*0.95)],
            'p99': sorted(durations)[int(len(durations)*0.99)],
            'std': self._calculate_std(durations),
            'total_time': sum(durations)
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def get_all_timing_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics for all operations."""
        return {op: self.get_timing_stats(op) for op in self.timing_data.keys()}
    
    def clear_timing_data(self, operation_name: Optional[str] = None):
        """Clear timing data for an operation or all operations."""
        if operation_name:
            if operation_name in self.timing_data:
                self.timing_data[operation_name].clear()
        else:
            self.timing_data.clear()


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    
    This class provides centralized performance monitoring with:
    - Metric collection and storage
    - Memory tracking and analysis
    - Timing profiling and statistics
    - Alert system for threshold violations
    - Regression detection and analysis
    """
    
    def __init__(self, 
                 enable_memory_tracking: bool = True,
                 memory_sample_interval: float = 1.0,
                 max_metrics: int = 100000,
                 alert_callbacks: Optional[List[Callable]] = None):
        """
        Initialize performance monitor.
        
        Parameters
        ----------
        enable_memory_tracking : bool
            Whether to enable background memory tracking
        memory_sample_interval : float
            Interval for memory sampling in seconds
        max_metrics : int
            Maximum number of metrics to store
        alert_callbacks : List[Callable], optional
            Callbacks to call when alerts are triggered
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Metric storage
        self.metrics: deque = deque(maxlen=max_metrics)
        self.metric_summaries: Dict[str, Dict[str, Any]] = {}
        
        # Components
        self.memory_tracker: Optional[MemoryTracker] = None
        if enable_memory_tracking:
            self.memory_tracker = MemoryTracker(memory_sample_interval)
        
        self.timing_profiler = TimingProfiler()
        
        # Alert system
        self.thresholds: Dict[str, PerformanceThreshold] = {}
        self.alerts: deque = deque(maxlen=1000)
        self.alert_callbacks = alert_callbacks or []
        
        # Performance baselines for regression detection
        self.baselines: Dict[str, Dict[str, float]] = {}
        
        # Monitoring state
        self.start_time = time.time()
        self.is_monitoring = True
        
        self.logger.info("Performance monitor initialized")
    
    def record_metric(self, 
                     name: str, 
                     value: float, 
                     metric_type: MetricType = MetricType.GAUGE,
                     tags: Optional[Dict[str, str]] = None,
                     description: Optional[str] = None):
        """Record a performance metric."""
        try:
            metric = PerformanceMetric(
                name=name,
                metric_type=metric_type,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
                description=description
            )
            
            self.metrics.append(metric)
            
            # Update summary statistics
            self._update_metric_summary(metric)
            
            # Check thresholds
            self._check_thresholds(metric)
            
        except Exception as e:
            self.logger.error(f"Failed to record metric {name}: {e}")
    
    def _update_metric_summary(self, metric: PerformanceMetric):
        """Update running summary statistics for a metric."""
        name = metric.name
        
        if name not in self.metric_summaries:
            self.metric_summaries[name] = {
                'count': 0,
                'sum': 0.0,
                'min': float('inf'),
                'max': float('-inf'),
                'last_value': 0.0,
                'last_timestamp': 0.0,
                'type': metric.metric_type.value
            }
        
        summary = self.metric_summaries[name]
        summary['count'] += 1
        summary['sum'] += metric.value
        summary['min'] = min(summary['min'], metric.value)
        summary['max'] = max(summary['max'], metric.value)
        summary['last_value'] = metric.value
        summary['last_timestamp'] = metric.timestamp
    
    def _check_thresholds(self, metric: PerformanceMetric):
        """Check if metric violates any thresholds."""
        if metric.name not in self.thresholds:
            return
        
        threshold = self.thresholds[metric.name]
        
        violated = False
        if threshold.comparison == "greater" and metric.value > threshold.threshold_value:
            violated = True
        elif threshold.comparison == "less" and metric.value < threshold.threshold_value:
            violated = True
        elif threshold.comparison == "equal" and metric.value == threshold.threshold_value:
            violated = True
        
        if violated:
            alert = PerformanceAlert(
                level=threshold.alert_level,
                metric_name=metric.name,
                message=f"Metric {metric.name} violated threshold: {metric.value} {threshold.comparison} {threshold.threshold_value}",
                value=metric.value,
                threshold=threshold.threshold_value,
                timestamp=metric.timestamp,
                tags=metric.tags
            )
            
            self._trigger_alert(alert)
    
    def _trigger_alert(self, alert: PerformanceAlert):
        """Trigger a performance alert."""
        self.alerts.append(alert)
        
        # Log alert
        log_func = getattr(self.logger, alert.level.value, self.logger.info)
        log_func(f"Performance alert: {alert.message}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def set_threshold(self, 
                     metric_name: str, 
                     threshold_value: float,
                     alert_level: AlertLevel = AlertLevel.WARNING,
                     comparison: str = "greater",
                     description: Optional[str] = None):
        """Set a performance threshold."""
        self.thresholds[metric_name] = PerformanceThreshold(
            metric_name=metric_name,
            threshold_value=threshold_value,
            alert_level=alert_level,
            comparison=comparison,
            description=description
        )
        
        self.logger.info(f"Set threshold for {metric_name}: {threshold_value} ({comparison})")
    
    def start_operation(self, operation_name: str, context: Optional[Dict[str, Any]] = None):
        """Start timing an operation."""
        self.timing_profiler.start_timer(operation_name, context)
    
    def end_operation(self, operation_name: str) -> float:
        """End timing an operation."""
        duration = self.timing_profiler.end_timer(operation_name)
        
        # Record as metric
        self.record_metric(
            f"{operation_name}_duration",
            duration,
            MetricType.TIMER,
            description=f"Duration of {operation_name} operation"
        )
        
        return duration
    
    def get_metric_summary(self, metric_name: str) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        if metric_name not in self.metric_summaries:
            return {}
        
        summary = self.metric_summaries[metric_name].copy()
        
        # Calculate average
        if summary['count'] > 0:
            summary['average'] = summary['sum'] / summary['count']
        else:
            summary['average'] = 0.0
        
        return summary
    
    def get_all_metrics_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics for all metrics."""
        return {name: self.get_metric_summary(name) for name in self.metric_summaries.keys()}
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if self.memory_tracker:
            return self.memory_tracker.get_memory_statistics()
        return {}
    
    def get_timing_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics for all operations."""
        return self.timing_profiler.get_all_timing_stats()
    
    def detect_regressions(self, metric_name: str, window_size: int = 100) -> Optional[Dict[str, Any]]:
        """Detect performance regressions for a metric."""
        if metric_name not in self.baselines:
            return None
        
        # Get recent metrics
        recent_metrics = [m for m in self.metrics if m.name == metric_name][-window_size:]
        
        if len(recent_metrics) < window_size // 2:
            return None
        
        # Calculate current performance
        current_values = [m.value for m in recent_metrics]
        current_avg = sum(current_values) / len(current_values)
        
        # Compare with baseline
        baseline = self.baselines[metric_name]
        baseline_avg = baseline.get('average', 0)
        
        if baseline_avg == 0:
            return None
        
        # Calculate regression percentage
        regression_pct = (current_avg - baseline_avg) / baseline_avg * 100
        
        # Detect significant regression (>20% worse)
        is_regression = regression_pct > 20
        
        return {
            'metric_name': metric_name,
            'current_average': current_avg,
            'baseline_average': baseline_avg,
            'regression_percent': regression_pct,
            'is_regression': is_regression,
            'sample_size': len(current_values)
        }
    
    def set_baseline(self, metric_name: str, window_size: int = 100):
        """Set performance baseline for a metric."""
        # Get recent metrics
        recent_metrics = [m for m in self.metrics if m.name == metric_name][-window_size:]
        
        if not recent_metrics:
            self.logger.warning(f"No metrics found for {metric_name}")
            return
        
        values = [m.value for m in recent_metrics]
        
        self.baselines[metric_name] = {
            'average': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'count': len(values),
            'timestamp': time.time()
        }
        
        self.logger.info(f"Set baseline for {metric_name} from {len(values)} samples")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        uptime = time.time() - self.start_time
        
        report = {
            'uptime_seconds': uptime,
            'total_metrics': len(self.metrics),
            'total_alerts': len(self.alerts),
            'metric_summaries': self.get_all_metrics_summary(),
            'timing_statistics': self.get_timing_statistics(),
            'memory_statistics': self.get_memory_statistics(),
            'active_thresholds': len(self.thresholds),
            'baselines': list(self.baselines.keys()),
            'recent_alerts': [
                {
                    'level': alert.level.value,
                    'metric': alert.metric_name,
                    'message': alert.message,
                    'timestamp': alert.timestamp
                }
                for alert in list(self.alerts)[-10:]  # Last 10 alerts
            ]
        }
        
        return report
    
    def export_metrics(self, file_path: str, format: str = "json"):
        """Export metrics to file."""
        try:
            data = {
                'metrics': [
                    {
                        'name': m.name,
                        'type': m.metric_type.value,
                        'value': m.value,
                        'timestamp': m.timestamp,
                        'tags': m.tags
                    }
                    for m in self.metrics
                ],
                'export_timestamp': time.time()
            }
            
            if format == "json":
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Exported {len(self.metrics)} metrics to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
    
    def clear_metrics(self):
        """Clear all stored metrics."""
        self.metrics.clear()
        self.metric_summaries.clear()
        self.alerts.clear()
        self.timing_profiler.clear_timing_data()
        self.logger.info("Cleared all performance metrics")
    
    def shutdown(self):
        """Shutdown the performance monitor."""
        self.is_monitoring = False
        
        if self.memory_tracker:
            self.memory_tracker.stop_monitoring()
        
        self.logger.info("Performance monitor shutdown")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()