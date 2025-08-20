"""
Advanced Scraper Scheduling System - 40by6 Implementation
Cron-based scheduling with priority queue and resource management
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import heapq
from croniter import croniter
import pytz
from enum import Enum

from .scraper_management_system import ScraperMetadata, ScraperStatus

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Types of schedules"""
    CONTINUOUS = "continuous"  # Run as soon as previous completes
    CRON = "cron"             # Cron expression
    INTERVAL = "interval"      # Fixed interval
    ONCE = "once"             # One-time execution
    ON_DEMAND = "on_demand"   # Manual trigger only


@dataclass
class ScheduledTask:
    """Scheduled scraper task"""
    scraper_id: str
    next_run: datetime
    schedule_type: ScheduleType
    schedule_expr: str
    priority: int
    retry_count: int = 0
    last_run: Optional[datetime] = None
    
    def __lt__(self, other):
        """Compare by next run time and priority"""
        if self.next_run == other.next_run:
            return self.priority > other.priority  # Higher priority first
        return self.next_run < other.next_run


class ResourceManager:
    """Manage system resources for scraper execution"""
    
    def __init__(self, max_concurrent: int = 20, max_cpu_percent: float = 80.0, max_memory_mb: int = 4096):
        self.max_concurrent = max_concurrent
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_mb = max_memory_mb
        self.running_scrapers: Set[str] = set()
        self.resource_usage: Dict[str, Dict[str, float]] = {}
    
    async def can_schedule(self, scraper: ScraperMetadata) -> bool:
        """Check if scraper can be scheduled based on resources"""
        if len(self.running_scrapers) >= self.max_concurrent:
            return False
        
        # Check CPU usage
        current_cpu = self._get_current_cpu_usage()
        if current_cpu > self.max_cpu_percent:
            logger.warning(f"CPU usage too high: {current_cpu}%")
            return False
        
        # Check memory usage
        current_memory = self._get_current_memory_usage()
        estimated_memory = self._estimate_scraper_memory(scraper)
        if current_memory + estimated_memory > self.max_memory_mb:
            logger.warning(f"Memory usage would exceed limit: {current_memory + estimated_memory}MB")
            return False
        
        # Check rate limits
        if not self._check_rate_limits(scraper):
            return False
        
        return True
    
    def _get_current_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0
    
    def _get_current_memory_usage(self) -> int:
        """Get current memory usage in MB"""
        try:
            import psutil
            return psutil.virtual_memory().used / 1024 / 1024
        except:
            return 0
    
    def _estimate_scraper_memory(self, scraper: ScraperMetadata) -> int:
        """Estimate memory usage for scraper"""
        # Base estimate by category
        category_estimates = {
            "federal_parliament": 512,
            "provincial_legislature": 256,
            "municipal_council": 128,
            "civic_platform": 256,
            "custom_scraper": 512
        }
        
        base = category_estimates.get(scraper.category.value, 256)
        
        # Adjust based on historical data
        if scraper.id in self.resource_usage:
            historical = self.resource_usage[scraper.id].get("memory_mb", base)
            return int(historical * 1.2)  # 20% buffer
        
        return base
    
    def _check_rate_limits(self, scraper: ScraperMetadata) -> bool:
        """Check if scraper respects rate limits"""
        if not scraper.rate_limit:
            return True
        
        # Check domain-specific rate limits
        domain = scraper.url.split('/')[2] if scraper.url else ""
        # Implementation would check Redis for recent requests to domain
        
        return True
    
    def register_start(self, scraper_id: str):
        """Register scraper start"""
        self.running_scrapers.add(scraper_id)
    
    def register_complete(self, scraper_id: str, metrics: Dict[str, float]):
        """Register scraper completion with metrics"""
        self.running_scrapers.discard(scraper_id)
        self.resource_usage[scraper_id] = metrics


class ScraperScheduler:
    """Advanced scheduler for scraper execution"""
    
    def __init__(self, timezone: str = "UTC"):
        self.timezone = pytz.timezone(timezone)
        self.schedule_queue: List[ScheduledTask] = []  # Min heap
        self.scrapers: Dict[str, ScraperMetadata] = {}
        self.resource_manager = ResourceManager()
        self.schedule_lock = asyncio.Lock()
        self.running = False
    
    def add_scraper(self, scraper: ScraperMetadata):
        """Add scraper to scheduler"""
        self.scrapers[scraper.id] = scraper
        
        # Parse schedule and create task
        schedule_type, schedule_expr = self._parse_schedule(scraper.schedule)
        
        if schedule_type != ScheduleType.ON_DEMAND:
            next_run = self._calculate_next_run(schedule_type, schedule_expr, scraper.last_run)
            
            task = ScheduledTask(
                scraper_id=scraper.id,
                next_run=next_run,
                schedule_type=schedule_type,
                schedule_expr=schedule_expr,
                priority=scraper.priority,
                last_run=scraper.last_run
            )
            
            heapq.heappush(self.schedule_queue, task)
            logger.info(f"Scheduled {scraper.name} for {next_run}")
    
    def _parse_schedule(self, schedule: str) -> Tuple[ScheduleType, str]:
        """Parse schedule string to determine type"""
        if not schedule:
            return ScheduleType.ON_DEMAND, ""
        
        schedule = schedule.strip()
        
        if schedule == "continuous":
            return ScheduleType.CONTINUOUS, ""
        elif schedule == "once":
            return ScheduleType.ONCE, ""
        elif schedule.startswith("every "):
            # Parse interval like "every 30 minutes"
            return ScheduleType.INTERVAL, schedule[6:]
        elif any(char in schedule for char in ["*", "?", "-", ","]):
            # Likely cron expression
            return ScheduleType.CRON, schedule
        else:
            return ScheduleType.ON_DEMAND, ""
    
    def _calculate_next_run(
        self, 
        schedule_type: ScheduleType, 
        schedule_expr: str,
        last_run: Optional[datetime]
    ) -> datetime:
        """Calculate next run time based on schedule"""
        now = datetime.now(self.timezone)
        
        if schedule_type == ScheduleType.CONTINUOUS:
            # Run immediately if not running
            return now
        
        elif schedule_type == ScheduleType.ONCE:
            # Only run if never run before
            return now if last_run is None else datetime.max
        
        elif schedule_type == ScheduleType.INTERVAL:
            # Parse interval
            interval = self._parse_interval(schedule_expr)
            if last_run:
                return last_run + interval
            else:
                return now
        
        elif schedule_type == ScheduleType.CRON:
            # Use croniter to calculate next run
            try:
                cron = croniter(schedule_expr, now)
                return cron.get_next(datetime)
            except Exception as e:
                logger.error(f"Invalid cron expression '{schedule_expr}': {e}")
                return now + timedelta(hours=24)  # Default to daily
        
        return datetime.max  # ON_DEMAND
    
    def _parse_interval(self, interval_str: str) -> timedelta:
        """Parse interval string to timedelta"""
        parts = interval_str.lower().split()
        if len(parts) != 2:
            return timedelta(hours=24)
        
        try:
            value = int(parts[0])
            unit = parts[1].rstrip('s')  # Remove plural 's'
            
            if unit == "minute":
                return timedelta(minutes=value)
            elif unit == "hour":
                return timedelta(hours=value)
            elif unit == "day":
                return timedelta(days=value)
            elif unit == "week":
                return timedelta(weeks=value)
            else:
                return timedelta(hours=24)
        except:
            return timedelta(hours=24)
    
    async def start(self):
        """Start the scheduler"""
        self.running = True
        logger.info("Scraper scheduler started")
        
        while self.running:
            try:
                await self._process_schedule()
                await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)
    
    async def stop(self):
        """Stop the scheduler"""
        self.running = False
        logger.info("Scraper scheduler stopped")
    
    async def _process_schedule(self):
        """Process scheduled tasks"""
        now = datetime.now(self.timezone)
        tasks_to_execute = []
        
        async with self.schedule_lock:
            # Get all due tasks
            while self.schedule_queue and self.schedule_queue[0].next_run <= now:
                task = heapq.heappop(self.schedule_queue)
                
                # Check if scraper still exists and is active
                if task.scraper_id in self.scrapers:
                    scraper = self.scrapers[task.scraper_id]
                    if scraper.status == ScraperStatus.ACTIVE:
                        tasks_to_execute.append((task, scraper))
                    else:
                        logger.info(f"Skipping inactive scraper {scraper.name}")
        
        # Execute tasks based on resource availability
        for task, scraper in tasks_to_execute:
            if await self.resource_manager.can_schedule(scraper):
                await self._execute_task(task, scraper)
            else:
                # Reschedule for later
                task.next_run = now + timedelta(minutes=5)
                task.retry_count += 1
                
                if task.retry_count < 10:  # Max retries
                    async with self.schedule_lock:
                        heapq.heappush(self.schedule_queue, task)
                    logger.info(f"Rescheduled {scraper.name} due to resource constraints")
    
    async def _execute_task(self, task: ScheduledTask, scraper: ScraperMetadata):
        """Execute a scheduled task"""
        logger.info(f"Executing scheduled task for {scraper.name}")
        
        # Register with resource manager
        self.resource_manager.register_start(scraper.id)
        
        # Send to execution queue (would integrate with orchestrator)
        # For now, just log
        logger.info(f"Would execute scraper {scraper.id}")
        
        # Schedule next run
        if task.schedule_type != ScheduleType.ONCE:
            task.last_run = datetime.now(self.timezone)
            task.next_run = self._calculate_next_run(
                task.schedule_type,
                task.schedule_expr,
                task.last_run
            )
            task.retry_count = 0
            
            async with self.schedule_lock:
                heapq.heappush(self.schedule_queue, task)
    
    def get_schedule_preview(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get preview of upcoming scheduled runs"""
        cutoff = datetime.now(self.timezone) + timedelta(hours=hours)
        preview = []
        
        # Create a copy of the queue to preview without modifying
        temp_queue = list(self.schedule_queue)
        heapq.heapify(temp_queue)
        
        while temp_queue and temp_queue[0].next_run <= cutoff:
            task = heapq.heappop(temp_queue)
            if task.scraper_id in self.scrapers:
                scraper = self.scrapers[task.scraper_id]
                preview.append({
                    "scraper_id": task.scraper_id,
                    "scraper_name": scraper.name,
                    "next_run": task.next_run.isoformat(),
                    "schedule_type": task.schedule_type.value,
                    "priority": task.priority
                })
        
        return sorted(preview, key=lambda x: x["next_run"])
    
    def update_schedule(self, scraper_id: str, new_schedule: str):
        """Update scraper schedule"""
        if scraper_id not in self.scrapers:
            return
        
        # Remove existing scheduled tasks
        async with self.schedule_lock:
            self.schedule_queue = [
                task for task in self.schedule_queue 
                if task.scraper_id != scraper_id
            ]
            heapq.heapify(self.schedule_queue)
        
        # Update scraper and reschedule
        scraper = self.scrapers[scraper_id]
        scraper.schedule = new_schedule
        self.add_scraper(scraper)
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        stats = {
            "total_scheduled": len(self.schedule_queue),
            "running_scrapers": len(self.resource_manager.running_scrapers),
            "resource_usage": {
                "cpu_percent": self.resource_manager._get_current_cpu_usage(),
                "memory_mb": self.resource_manager._get_current_memory_usage(),
                "concurrent_limit": self.resource_manager.max_concurrent
            },
            "schedule_types": {},
            "next_24h_count": 0
        }
        
        # Count by schedule type
        for task in self.schedule_queue:
            stype = task.schedule_type.value
            stats["schedule_types"][stype] = stats["schedule_types"].get(stype, 0) + 1
        
        # Count runs in next 24 hours
        cutoff = datetime.now(self.timezone) + timedelta(hours=24)
        stats["next_24h_count"] = sum(1 for task in self.schedule_queue if task.next_run <= cutoff)
        
        return stats


class SmartScheduler:
    """Smart scheduler that learns optimal execution times"""
    
    def __init__(self, scheduler: ScraperScheduler):
        self.scheduler = scheduler
        self.execution_history: Dict[str, List[Dict[str, Any]]] = {}
        self.optimization_enabled = True
    
    async def analyze_performance(self, scraper_id: str) -> Dict[str, Any]:
        """Analyze scraper performance to optimize schedule"""
        if scraper_id not in self.execution_history:
            return {}
        
        history = self.execution_history[scraper_id]
        if len(history) < 10:  # Need enough data
            return {}
        
        analysis = {
            "optimal_time": self._find_optimal_time(history),
            "average_duration": self._calculate_average_duration(history),
            "success_by_hour": self._analyze_success_by_hour(history),
            "recommendations": []
        }
        
        # Generate recommendations
        if analysis["optimal_time"]:
            analysis["recommendations"].append(
                f"Schedule at {analysis['optimal_time']} for best success rate"
            )
        
        return analysis
    
    def _find_optimal_time(self, history: List[Dict[str, Any]]) -> Optional[str]:
        """Find optimal execution time based on success rates"""
        hour_success = {}
        
        for run in history:
            hour = run["start_time"].hour
            success = run["status"] == "success"
            
            if hour not in hour_success:
                hour_success[hour] = {"success": 0, "total": 0}
            
            hour_success[hour]["total"] += 1
            if success:
                hour_success[hour]["success"] += 1
        
        # Find hour with best success rate
        best_hour = None
        best_rate = 0
        
        for hour, stats in hour_success.items():
            rate = stats["success"] / stats["total"]
            if rate > best_rate and stats["total"] >= 3:  # Minimum runs
                best_rate = rate
                best_hour = hour
        
        return f"{best_hour:02d}:00" if best_hour is not None else None
    
    def _calculate_average_duration(self, history: List[Dict[str, Any]]) -> float:
        """Calculate average execution duration"""
        durations = [
            run["duration"] for run in history 
            if "duration" in run and run["status"] == "success"
        ]
        
        return sum(durations) / len(durations) if durations else 0
    
    def _analyze_success_by_hour(self, history: List[Dict[str, Any]]) -> Dict[int, float]:
        """Analyze success rate by hour of day"""
        hour_stats = {}
        
        for run in history:
            hour = run["start_time"].hour
            if hour not in hour_stats:
                hour_stats[hour] = {"success": 0, "total": 0}
            
            hour_stats[hour]["total"] += 1
            if run["status"] == "success":
                hour_stats[hour]["success"] += 1
        
        return {
            hour: stats["success"] / stats["total"] 
            for hour, stats in hour_stats.items()
            if stats["total"] > 0
        }
    
    def record_execution(self, scraper_id: str, execution_data: Dict[str, Any]):
        """Record execution for analysis"""
        if scraper_id not in self.execution_history:
            self.execution_history[scraper_id] = []
        
        self.execution_history[scraper_id].append(execution_data)
        
        # Keep only recent history
        if len(self.execution_history[scraper_id]) > 100:
            self.execution_history[scraper_id] = self.execution_history[scraper_id][-100:]
    
    async def optimize_schedules(self):
        """Optimize schedules based on performance data"""
        if not self.optimization_enabled:
            return
        
        for scraper_id in self.scheduler.scrapers:
            analysis = await self.analyze_performance(scraper_id)
            
            if analysis.get("optimal_time") and analysis.get("recommendations"):
                logger.info(f"Schedule optimization for {scraper_id}: {analysis['recommendations']}")
                # Could automatically update schedule here