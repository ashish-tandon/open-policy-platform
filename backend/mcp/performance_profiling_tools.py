"""
Performance Profiling and Optimization Tools - 40by6
Advanced performance monitoring, profiling, and optimization
"""

import asyncio
import logging
import time
import psutil
import cProfile
import pstats
import tracemalloc
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import threading
import multiprocessing
import aiohttp
import redis.asyncio as redis
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import yaml
from pathlib import Path
import sys
import resource
import signal
from contextlib import contextmanager
from functools import wraps
import inspect

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"
    API_LATENCY = "api_latency"
    SCRAPER_PERFORMANCE = "scraper_performance"
    CUSTOM = "custom"


@dataclass
class PerformanceMetric:
    """Performance metric data"""
    name: str
    type: MetricType
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.type.value,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags,
            'metadata': self.metadata
        }


@dataclass
class ProfileResult:
    """Result of profiling session"""
    profile_id: str
    start_time: datetime
    end_time: datetime
    duration: float
    metrics: List[PerformanceMetric]
    cpu_profile: Optional[Dict[str, Any]] = None
    memory_profile: Optional[Dict[str, Any]] = None
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'profile_id': self.profile_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration': self.duration,
            'metrics': [m.to_dict() for m in self.metrics],
            'cpu_profile': self.cpu_profile,
            'memory_profile': self.memory_profile,
            'bottlenecks': self.bottlenecks,
            'recommendations': self.recommendations
        }


class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self, redis_url: str = 'redis://localhost:6379'):
        self.redis_url = redis_url
        self.redis_client = None
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.monitoring_tasks = []
        self.is_monitoring = False
        
    async def initialize(self):
        """Initialize performance monitor"""
        self.redis_client = await redis.from_url(self.redis_url)
        logger.info("Performance monitor initialized")
    
    async def start_monitoring(self, interval: int = 5):
        """Start monitoring system performance"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        
        # Start monitoring tasks
        tasks = [
            self._monitor_system_resources(interval),
            self._monitor_database_performance(interval),
            self._monitor_cache_performance(interval),
            self._monitor_queue_performance(interval)
        ]
        
        for task in tasks:
            self.monitoring_tasks.append(asyncio.create_task(task))
        
        logger.info(f"Started performance monitoring with {interval}s interval")
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        
        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        self.monitoring_tasks.clear()
        
        logger.info("Stopped performance monitoring")
    
    async def _monitor_system_resources(self, interval: int):
        """Monitor system resources"""
        while self.is_monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                await self._record_metric(PerformanceMetric(
                    name='system.cpu.usage',
                    type=MetricType.CPU,
                    value=cpu_percent,
                    unit='percent'
                ))
                
                # CPU per core
                cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
                for i, percent in enumerate(cpu_per_core):
                    await self._record_metric(PerformanceMetric(
                        name=f'system.cpu.core{i}',
                        type=MetricType.CPU,
                        value=percent,
                        unit='percent',
                        tags={'core': str(i)}
                    ))
                
                # Memory usage
                memory = psutil.virtual_memory()
                await self._record_metric(PerformanceMetric(
                    name='system.memory.used',
                    type=MetricType.MEMORY,
                    value=memory.used,
                    unit='bytes',
                    metadata={
                        'total': memory.total,
                        'available': memory.available,
                        'percent': memory.percent
                    }
                ))
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    await self._record_metric(PerformanceMetric(
                        name='system.disk.read_bytes',
                        type=MetricType.DISK_IO,
                        value=disk_io.read_bytes,
                        unit='bytes'
                    ))
                    await self._record_metric(PerformanceMetric(
                        name='system.disk.write_bytes',
                        type=MetricType.DISK_IO,
                        value=disk_io.write_bytes,
                        unit='bytes'
                    ))
                
                # Network I/O
                net_io = psutil.net_io_counters()
                await self._record_metric(PerformanceMetric(
                    name='system.network.bytes_sent',
                    type=MetricType.NETWORK_IO,
                    value=net_io.bytes_sent,
                    unit='bytes'
                ))
                await self._record_metric(PerformanceMetric(
                    name='system.network.bytes_recv',
                    type=MetricType.NETWORK_IO,
                    value=net_io.bytes_recv,
                    unit='bytes'
                ))
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error monitoring system resources: {e}")
                await asyncio.sleep(interval)
    
    async def _monitor_database_performance(self, interval: int):
        """Monitor database performance"""
        while self.is_monitoring:
            try:
                # This would connect to actual database
                # For demo, using mock data
                
                # Query execution time
                await self._record_metric(PerformanceMetric(
                    name='database.query.avg_time',
                    type=MetricType.DATABASE,
                    value=np.random.uniform(0.001, 0.1),
                    unit='seconds'
                ))
                
                # Connection pool stats
                await self._record_metric(PerformanceMetric(
                    name='database.connections.active',
                    type=MetricType.DATABASE,
                    value=np.random.randint(5, 20),
                    unit='count'
                ))
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error monitoring database: {e}")
                await asyncio.sleep(interval)
    
    async def _monitor_cache_performance(self, interval: int):
        """Monitor cache performance"""
        while self.is_monitoring:
            try:
                if self.redis_client:
                    # Get Redis info
                    info = await self.redis_client.info()
                    
                    # Cache hit ratio
                    hits = info.get('keyspace_hits', 0)
                    misses = info.get('keyspace_misses', 0)
                    total = hits + misses
                    hit_ratio = (hits / total * 100) if total > 0 else 0
                    
                    await self._record_metric(PerformanceMetric(
                        name='cache.hit_ratio',
                        type=MetricType.CACHE,
                        value=hit_ratio,
                        unit='percent',
                        metadata={'hits': hits, 'misses': misses}
                    ))
                    
                    # Memory usage
                    await self._record_metric(PerformanceMetric(
                        name='cache.memory.used',
                        type=MetricType.CACHE,
                        value=info.get('used_memory', 0),
                        unit='bytes'
                    ))
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error monitoring cache: {e}")
                await asyncio.sleep(interval)
    
    async def _monitor_queue_performance(self, interval: int):
        """Monitor queue performance"""
        while self.is_monitoring:
            try:
                if self.redis_client:
                    # Queue sizes
                    queues = ['scraper:queue', 'scraper:ingestion:queue', 'scraper:priority:queue']
                    
                    for queue_name in queues:
                        size = await self.redis_client.llen(queue_name)
                        
                        await self._record_metric(PerformanceMetric(
                            name=f'queue.{queue_name}.size',
                            type=MetricType.QUEUE,
                            value=size,
                            unit='items',
                            tags={'queue': queue_name}
                        ))
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error monitoring queues: {e}")
                await asyncio.sleep(interval)
    
    async def _record_metric(self, metric: PerformanceMetric):
        """Record a performance metric"""
        # Add to buffer
        self.metrics_buffer[metric.name].append(metric)
        
        # Publish to Redis
        if self.redis_client:
            await self.redis_client.publish(
                'performance:metrics',
                json.dumps(metric.to_dict())
            )
            
            # Store in time series
            await self.redis_client.zadd(
                f'metrics:{metric.name}',
                {json.dumps(metric.to_dict()): metric.timestamp.timestamp()}
            )
            
            # Expire old data (keep 24 hours)
            cutoff = (datetime.utcnow() - timedelta(hours=24)).timestamp()
            await self.redis_client.zremrangebyscore(f'metrics:{metric.name}', 0, cutoff)
    
    async def get_metrics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[PerformanceMetric]:
        """Get metrics for time range"""
        
        if self.redis_client:
            data = await self.redis_client.zrangebyscore(
                f'metrics:{metric_name}',
                start_time.timestamp(),
                end_time.timestamp()
            )
            
            metrics = []
            for item in data:
                metric_data = json.loads(item)
                metrics.append(PerformanceMetric(
                    name=metric_data['name'],
                    type=MetricType(metric_data['type']),
                    value=metric_data['value'],
                    unit=metric_data['unit'],
                    timestamp=datetime.fromisoformat(metric_data['timestamp']),
                    tags=metric_data.get('tags', {}),
                    metadata=metric_data.get('metadata', {})
                ))
            
            return metrics
        
        return []
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance stats"""
        stats = {}
        
        for metric_name, buffer in self.metrics_buffer.items():
            if buffer:
                recent_values = [m.value for m in list(buffer)[-10:]]
                stats[metric_name] = {
                    'current': buffer[-1].value,
                    'avg': np.mean(recent_values),
                    'min': np.min(recent_values),
                    'max': np.max(recent_values),
                    'unit': buffer[-1].unit
                }
        
        return stats


class CPUProfiler:
    """CPU profiling utilities"""
    
    def __init__(self):
        self.profiler = cProfile.Profile()
        self.is_profiling = False
        
    @contextmanager
    def profile(self, sort_by: str = 'cumulative'):
        """Context manager for CPU profiling"""
        self.profiler.enable()
        self.is_profiling = True
        
        try:
            yield self
        finally:
            self.profiler.disable()
            self.is_profiling = False
    
    def get_stats(self, top_n: int = 20) -> Dict[str, Any]:
        """Get profiling statistics"""
        if self.is_profiling:
            self.profiler.disable()
        
        stats = pstats.Stats(self.profiler)
        stats.sort_stats('cumulative')
        
        # Get top functions
        top_functions = []
        for func, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:top_n]:
            top_functions.append({
                'function': f"{func[0]}:{func[1]}:{func[2]}",
                'calls': nc,
                'total_time': tt,
                'cumulative_time': ct,
                'time_per_call': tt/nc if nc > 0 else 0
            })
        
        return {
            'total_calls': sum(s[1][1] for s in stats.stats.items()),
            'total_time': sum(s[1][2] for s in stats.stats.items()),
            'top_functions': top_functions
        }
    
    def print_stats(self, top_n: int = 20):
        """Print profiling statistics"""
        stats = pstats.Stats(self.profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(top_n)


class MemoryProfiler:
    """Memory profiling utilities"""
    
    def __init__(self):
        self.snapshots = []
        self.is_tracing = False
    
    def start_tracing(self):
        """Start memory tracing"""
        if not self.is_tracing:
            tracemalloc.start()
            self.is_tracing = True
            self.snapshots.append(tracemalloc.take_snapshot())
    
    def stop_tracing(self):
        """Stop memory tracing"""
        if self.is_tracing:
            tracemalloc.stop()
            self.is_tracing = False
    
    def take_snapshot(self) -> Dict[str, Any]:
        """Take memory snapshot"""
        if not self.is_tracing:
            return {}
        
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append(snapshot)
        
        # Get top memory allocations
        top_stats = snapshot.statistics('lineno')[:20]
        
        allocations = []
        for stat in top_stats:
            allocations.append({
                'file': stat.traceback.format()[0] if stat.traceback else 'unknown',
                'size': stat.size,
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count
            })
        
        # Memory usage
        current, peak = tracemalloc.get_traced_memory()
        
        return {
            'current_mb': current / 1024 / 1024,
            'peak_mb': peak / 1024 / 1024,
            'top_allocations': allocations,
            'gc_stats': gc.get_stats()
        }
    
    def compare_snapshots(self, snapshot1_idx: int = 0, snapshot2_idx: int = -1) -> Dict[str, Any]:
        """Compare two memory snapshots"""
        if len(self.snapshots) < 2:
            return {}
        
        snapshot1 = self.snapshots[snapshot1_idx]
        snapshot2 = self.snapshots[snapshot2_idx]
        
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')[:20]
        
        differences = []
        for stat in top_stats:
            differences.append({
                'file': stat.traceback.format()[0] if stat.traceback else 'unknown',
                'size_diff': stat.size_diff,
                'size_diff_mb': stat.size_diff / 1024 / 1024,
                'count_diff': stat.count_diff
            })
        
        return {
            'total_diff_mb': sum(s.size_diff for s in top_stats) / 1024 / 1024,
            'differences': differences
        }


class ScraperPerformanceAnalyzer:
    """Analyze scraper-specific performance"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
    
    async def analyze_scraper_performance(
        self,
        scraper_id: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """Analyze performance of a specific scraper"""
        
        session = self.Session()
        try:
            # Get scraper runs
            since = datetime.utcnow() - timedelta(days=days)
            
            query = text("""
                SELECT 
                    start_time,
                    end_time,
                    status,
                    records_scraped,
                    data_quality_score,
                    errors
                FROM scraper_runs
                WHERE scraper_id = :scraper_id
                    AND start_time > :since
                ORDER BY start_time
            """)
            
            results = session.execute(query, {
                'scraper_id': scraper_id,
                'since': since
            }).fetchall()
            
            if not results:
                return {'error': 'No data available'}
            
            # Calculate metrics
            durations = []
            success_count = 0
            records_per_run = []
            quality_scores = []
            
            for row in results:
                if row.end_time and row.start_time:
                    duration = (row.end_time - row.start_time).total_seconds()
                    durations.append(duration)
                
                if row.status == 'completed':
                    success_count += 1
                    records_per_run.append(row.records_scraped or 0)
                    if row.data_quality_score:
                        quality_scores.append(row.data_quality_score)
            
            # Analysis results
            analysis = {
                'scraper_id': scraper_id,
                'period': f'{days} days',
                'total_runs': len(results),
                'success_rate': (success_count / len(results)) * 100 if results else 0,
                'performance': {
                    'avg_duration': np.mean(durations) if durations else 0,
                    'min_duration': np.min(durations) if durations else 0,
                    'max_duration': np.max(durations) if durations else 0,
                    'p95_duration': np.percentile(durations, 95) if durations else 0
                },
                'throughput': {
                    'avg_records': np.mean(records_per_run) if records_per_run else 0,
                    'total_records': sum(records_per_run),
                    'records_per_second': sum(records_per_run) / sum(durations) if durations else 0
                },
                'quality': {
                    'avg_score': np.mean(quality_scores) if quality_scores else 0,
                    'min_score': np.min(quality_scores) if quality_scores else 0,
                    'max_score': np.max(quality_scores) if quality_scores else 0
                },
                'trends': self._calculate_trends(results),
                'bottlenecks': self._identify_bottlenecks(results, durations),
                'recommendations': []
            }
            
            # Generate recommendations
            analysis['recommendations'] = self._generate_recommendations(analysis)
            
            return analysis
            
        finally:
            session.close()
    
    def _calculate_trends(self, results) -> Dict[str, Any]:
        """Calculate performance trends"""
        if len(results) < 2:
            return {}
        
        # Group by day
        daily_stats = defaultdict(list)
        
        for row in results:
            if row.start_time and row.end_time:
                day = row.start_time.date()
                duration = (row.end_time - row.start_time).total_seconds()
                daily_stats[day].append({
                    'duration': duration,
                    'records': row.records_scraped or 0,
                    'success': row.status == 'completed'
                })
        
        # Calculate daily averages
        trends = []
        for day, stats in sorted(daily_stats.items()):
            durations = [s['duration'] for s in stats]
            records = [s['records'] for s in stats if s['success']]
            
            trends.append({
                'date': day.isoformat(),
                'avg_duration': np.mean(durations) if durations else 0,
                'total_records': sum(records),
                'success_rate': sum(1 for s in stats if s['success']) / len(stats) * 100
            })
        
        return {
            'daily': trends,
            'direction': self._calculate_trend_direction(trends)
        }
    
    def _calculate_trend_direction(self, trends: List[Dict[str, Any]]) -> str:
        """Calculate overall trend direction"""
        if len(trends) < 2:
            return 'stable'
        
        # Compare first half to second half
        mid = len(trends) // 2
        first_half_avg = np.mean([t['avg_duration'] for t in trends[:mid]])
        second_half_avg = np.mean([t['avg_duration'] for t in trends[mid:]])
        
        change = (second_half_avg - first_half_avg) / first_half_avg if first_half_avg > 0 else 0
        
        if change > 0.1:
            return 'degrading'
        elif change < -0.1:
            return 'improving'
        else:
            return 'stable'
    
    def _identify_bottlenecks(self, results, durations: List[float]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        if not durations:
            return bottlenecks
        
        # Slow runs (> 95th percentile)
        p95 = np.percentile(durations, 95)
        slow_runs = sum(1 for d in durations if d > p95)
        
        if slow_runs > len(durations) * 0.1:  # More than 10% are slow
            bottlenecks.append({
                'type': 'slow_execution',
                'severity': 'high',
                'description': f'{slow_runs} runs exceeded 95th percentile ({p95:.1f}s)',
                'impact': 'reduced_throughput'
            })
        
        # High failure rate
        failure_rate = sum(1 for r in results if r.status != 'completed') / len(results)
        if failure_rate > 0.1:
            bottlenecks.append({
                'type': 'high_failure_rate',
                'severity': 'critical' if failure_rate > 0.3 else 'high',
                'description': f'Failure rate is {failure_rate*100:.1f}%',
                'impact': 'reduced_reliability'
            })
        
        # Memory issues (check error logs)
        memory_errors = sum(1 for r in results if r.errors and 'memory' in str(r.errors).lower())
        if memory_errors > 0:
            bottlenecks.append({
                'type': 'memory_issues',
                'severity': 'medium',
                'description': f'{memory_errors} runs had memory-related errors',
                'impact': 'potential_crashes'
            })
        
        return bottlenecks
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # Based on duration
        if analysis['performance']['avg_duration'] > 60:
            recommendations.append("Consider implementing pagination or chunking for large datasets")
        
        if analysis['performance']['p95_duration'] > analysis['performance']['avg_duration'] * 2:
            recommendations.append("Investigate causes of performance variance - possible rate limiting or network issues")
        
        # Based on throughput
        if analysis['throughput']['records_per_second'] < 10:
            recommendations.append("Optimize data extraction logic - consider parallel processing")
        
        # Based on quality
        if analysis['quality']['avg_score'] < 0.8:
            recommendations.append("Improve data validation and cleaning logic")
        
        # Based on bottlenecks
        for bottleneck in analysis['bottlenecks']:
            if bottleneck['type'] == 'slow_execution':
                recommendations.append("Profile slow runs to identify performance bottlenecks")
            elif bottleneck['type'] == 'high_failure_rate':
                recommendations.append("Implement better error handling and retry logic")
            elif bottleneck['type'] == 'memory_issues':
                recommendations.append("Optimize memory usage - process data in streams rather than loading all at once")
        
        # Based on trends
        if analysis['trends'].get('direction') == 'degrading':
            recommendations.append("Performance is degrading over time - check for data growth or external API changes")
        
        return recommendations


class PerformanceOptimizer:
    """Automated performance optimization"""
    
    def __init__(self):
        self.optimization_rules = self._load_optimization_rules()
        
    def _load_optimization_rules(self) -> List[Dict[str, Any]]:
        """Load optimization rules"""
        return [
            {
                'name': 'database_connection_pooling',
                'condition': lambda m: m.get('database.connections.waiting', 0) > 10,
                'action': 'increase_connection_pool_size',
                'description': 'Increase database connection pool when many waiting connections'
            },
            {
                'name': 'cache_optimization',
                'condition': lambda m: m.get('cache.hit_ratio', 100) < 80,
                'action': 'review_cache_strategy',
                'description': 'Cache hit ratio below 80% - review caching strategy'
            },
            {
                'name': 'memory_pressure',
                'condition': lambda m: m.get('system.memory.percent', 0) > 85,
                'action': 'trigger_garbage_collection',
                'description': 'High memory usage - trigger garbage collection'
            },
            {
                'name': 'queue_backlog',
                'condition': lambda m: any(v > 1000 for k, v in m.items() if k.startswith('queue.') and k.endswith('.size')),
                'action': 'scale_workers',
                'description': 'Queue backlog detected - scale up workers'
            }
        ]
    
    async def analyze_and_optimize(self, current_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze metrics and suggest optimizations"""
        optimizations = []
        
        # Flatten metrics for rule evaluation
        flat_metrics = {}
        for key, value in current_metrics.items():
            if isinstance(value, dict):
                flat_metrics[key] = value.get('current', value.get('avg', 0))
            else:
                flat_metrics[key] = value
        
        # Check optimization rules
        for rule in self.optimization_rules:
            if rule['condition'](flat_metrics):
                optimizations.append({
                    'rule': rule['name'],
                    'action': rule['action'],
                    'description': rule['description'],
                    'priority': self._calculate_priority(rule, flat_metrics)
                })
        
        # Sort by priority
        optimizations.sort(key=lambda x: x['priority'], reverse=True)
        
        return optimizations
    
    def _calculate_priority(self, rule: Dict[str, Any], metrics: Dict[str, float]) -> int:
        """Calculate optimization priority"""
        # Simple priority based on rule type
        priorities = {
            'memory_pressure': 100,
            'database_connection_pooling': 80,
            'queue_backlog': 70,
            'cache_optimization': 50
        }
        
        return priorities.get(rule['name'], 50)
    
    async def apply_optimization(self, optimization: Dict[str, Any]) -> bool:
        """Apply optimization action"""
        action = optimization['action']
        
        if action == 'trigger_garbage_collection':
            gc.collect()
            logger.info("Triggered garbage collection")
            return True
        
        elif action == 'increase_connection_pool_size':
            # This would adjust database connection pool
            logger.info("Would increase database connection pool size")
            return True
        
        elif action == 'scale_workers':
            # This would scale worker processes
            logger.info("Would scale up worker processes")
            return True
        
        else:
            logger.info(f"Optimization action not implemented: {action}")
            return False


def performance_decorator(metric_name: str = None):
    """Decorator for measuring function performance"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Get function name if not provided
            name = metric_name or f"function.{func.__name__}"
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record metric
                metric = PerformanceMetric(
                    name=name,
                    type=MetricType.CUSTOM,
                    value=duration,
                    unit='seconds',
                    tags={'function': func.__name__, 'status': 'success'}
                )
                
                # Log if slow
                if duration > 1.0:
                    logger.warning(f"Slow function execution: {name} took {duration:.2f}s")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Record error metric
                metric = PerformanceMetric(
                    name=name,
                    type=MetricType.CUSTOM,
                    value=duration,
                    unit='seconds',
                    tags={'function': func.__name__, 'status': 'error', 'error': str(e)}
                )
                
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            name = metric_name or f"function.{func.__name__}"
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                if duration > 1.0:
                    logger.warning(f"Slow function execution: {name} took {duration:.2f}s")
                
                return result
                
            except Exception as e:
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


class PerformanceReport:
    """Generate performance reports"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
    
    async def generate_report(
        self,
        start_time: datetime,
        end_time: datetime,
        include_graphs: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        report = {
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'duration_hours': (end_time - start_time).total_seconds() / 3600
            },
            'summary': {},
            'metrics': {},
            'visualizations': []
        }
        
        # Get all metric names
        metric_names = [
            'system.cpu.usage',
            'system.memory.used',
            'database.query.avg_time',
            'cache.hit_ratio',
            'queue.scraper:queue.size'
        ]
        
        for metric_name in metric_names:
            metrics = await self.monitor.get_metrics(metric_name, start_time, end_time)
            
            if metrics:
                values = [m.value for m in metrics]
                
                report['metrics'][metric_name] = {
                    'count': len(metrics),
                    'avg': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'p50': np.percentile(values, 50),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99),
                    'unit': metrics[0].unit
                }
                
                if include_graphs:
                    viz = self._create_metric_visualization(metrics, metric_name)
                    report['visualizations'].append(viz)
        
        # Calculate summary
        report['summary'] = self._calculate_summary(report['metrics'])
        
        return report
    
    def _create_metric_visualization(
        self,
        metrics: List[PerformanceMetric],
        metric_name: str
    ) -> Dict[str, Any]:
        """Create visualization for metric"""
        
        timestamps = [m.timestamp for m in metrics]
        values = [m.value for m in metrics]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='lines',
            name=metric_name,
            line=dict(width=2)
        ))
        
        fig.update_layout(
            title=f"{metric_name} Over Time",
            xaxis_title="Time",
            yaxis_title=metrics[0].unit if metrics else "Value",
            height=400,
            showlegend=False
        )
        
        return {
            'type': 'time_series',
            'metric': metric_name,
            'data': fig.to_dict()
        }
    
    def _calculate_summary(self, metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        
        summary = {
            'performance_score': 100,
            'issues': [],
            'highlights': []
        }
        
        # Check CPU usage
        if 'system.cpu.usage' in metrics:
            cpu_avg = metrics['system.cpu.usage']['avg']
            if cpu_avg > 80:
                summary['performance_score'] -= 20
                summary['issues'].append(f"High CPU usage: {cpu_avg:.1f}%")
            elif cpu_avg < 50:
                summary['highlights'].append(f"Efficient CPU usage: {cpu_avg:.1f}%")
        
        # Check memory usage
        if 'system.memory.used' in metrics:
            memory_max = metrics['system.memory.used']['max']
            memory_max_gb = memory_max / 1024 / 1024 / 1024
            if memory_max_gb > 8:
                summary['performance_score'] -= 15
                summary['issues'].append(f"High memory usage: {memory_max_gb:.1f}GB")
        
        # Check database performance
        if 'database.query.avg_time' in metrics:
            db_p95 = metrics['database.query.avg_time']['p95']
            if db_p95 > 0.1:
                summary['performance_score'] -= 10
                summary['issues'].append(f"Slow database queries: p95={db_p95*1000:.0f}ms")
        
        # Check cache performance
        if 'cache.hit_ratio' in metrics:
            cache_avg = metrics['cache.hit_ratio']['avg']
            if cache_avg < 80:
                summary['performance_score'] -= 10
                summary['issues'].append(f"Low cache hit ratio: {cache_avg:.1f}%")
            else:
                summary['highlights'].append(f"Good cache performance: {cache_avg:.1f}% hit ratio")
        
        return summary


# Example usage
async def performance_demo():
    """Demo performance profiling"""
    
    # Initialize monitor
    monitor = PerformanceMonitor()
    await monitor.initialize()
    
    # Start monitoring
    await monitor.start_monitoring(interval=2)
    
    # CPU profiling example
    cpu_profiler = CPUProfiler()
    
    with cpu_profiler.profile():
        # Simulate some work
        for i in range(100000):
            _ = sum(j**2 for j in range(100))
    
    cpu_stats = cpu_profiler.get_stats()
    print(f"CPU Profile: {cpu_stats['total_calls']} calls in {cpu_stats['total_time']:.2f}s")
    
    # Memory profiling example
    memory_profiler = MemoryProfiler()
    memory_profiler.start_tracing()
    
    # Take initial snapshot
    snapshot1 = memory_profiler.take_snapshot()
    print(f"Initial memory: {snapshot1['current_mb']:.1f}MB")
    
    # Allocate some memory
    data = [list(range(1000)) for _ in range(1000)]
    
    # Take second snapshot
    snapshot2 = memory_profiler.take_snapshot()
    print(f"After allocation: {snapshot2['current_mb']:.1f}MB")
    
    # Compare snapshots
    diff = memory_profiler.compare_snapshots()
    print(f"Memory increase: {diff['total_diff_mb']:.1f}MB")
    
    memory_profiler.stop_tracing()
    
    # Get current stats
    await asyncio.sleep(5)  # Let monitor collect some data
    stats = monitor.get_current_stats()
    print("\nCurrent Performance Stats:")
    for metric, values in stats.items():
        print(f"  {metric}: {values['current']:.2f} {values['unit']}")
    
    # Optimization suggestions
    optimizer = PerformanceOptimizer()
    optimizations = await optimizer.analyze_and_optimize(stats)
    
    if optimizations:
        print("\nOptimization Suggestions:")
        for opt in optimizations:
            print(f"  - {opt['description']}")
    
    # Stop monitoring
    await monitor.stop_monitoring()
    
    # Generate report
    report_gen = PerformanceReport(monitor)
    report = await report_gen.generate_report(
        datetime.utcnow() - timedelta(minutes=5),
        datetime.utcnow()
    )
    
    print(f"\nPerformance Report Summary:")
    print(f"  Score: {report['summary']['performance_score']}/100")
    print(f"  Issues: {len(report['summary']['issues'])}")
    print(f"  Highlights: {len(report['summary']['highlights'])}")


if __name__ == "__main__":
    asyncio.run(performance_demo())