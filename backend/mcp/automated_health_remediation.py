"""
Automated Health Remediation System - 40by6
Automatically detects and fixes health issues in the scraper ecosystem
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import aiohttp
import redis.asyncio as redis
from collections import defaultdict
import subprocess
import os
import psutil
import docker
from abc import ABC, abstractmethod

from .scraper_management_system import ScraperMetadata, ScraperStatus

logger = logging.getLogger(__name__)


class HealthIssueType(Enum):
    """Types of health issues"""
    SCRAPER_FAILURE = "scraper_failure"
    HIGH_ERROR_RATE = "high_error_rate"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MEMORY_LEAK = "memory_leak"
    QUEUE_BACKUP = "queue_backup"
    DATABASE_CONNECTION = "database_connection"
    DISK_SPACE = "disk_space"
    NETWORK_CONNECTIVITY = "network_connectivity"
    CERTIFICATE_EXPIRY = "certificate_expiry"
    DATA_QUALITY = "data_quality"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"


class RemediationStatus(Enum):
    """Status of remediation attempts"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REQUIRES_MANUAL = "requires_manual"


@dataclass
class HealthIssue:
    """Represents a detected health issue"""
    id: str
    type: HealthIssueType
    severity: str  # critical, high, medium, low
    component: str
    description: str
    detected_at: datetime
    metrics: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type.value,
            'severity': self.severity,
            'component': self.component,
            'description': self.description,
            'detected_at': self.detected_at.isoformat(),
            'metrics': self.metrics,
            'context': self.context
        }


@dataclass
class RemediationAction:
    """Represents a remediation action"""
    issue_id: str
    action_type: str
    parameters: Dict[str, Any]
    status: RemediationStatus = RemediationStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    
    def can_retry(self) -> bool:
        return self.attempts < self.max_attempts and self.status != RemediationStatus.SUCCEEDED


class RemediationStrategy(ABC):
    """Base class for remediation strategies"""
    
    @abstractmethod
    async def can_handle(self, issue: HealthIssue) -> bool:
        """Check if this strategy can handle the issue"""
        pass
    
    @abstractmethod
    async def remediate(self, issue: HealthIssue, context: Dict[str, Any]) -> RemediationAction:
        """Execute remediation for the issue"""
        pass
    
    @abstractmethod
    def get_priority(self, issue: HealthIssue) -> int:
        """Get priority for this remediation (higher = more urgent)"""
        pass


class ScraperRestartStrategy(RemediationStrategy):
    """Strategy for restarting failed scrapers"""
    
    async def can_handle(self, issue: HealthIssue) -> bool:
        return issue.type in [
            HealthIssueType.SCRAPER_FAILURE,
            HealthIssueType.MEMORY_LEAK,
            HealthIssueType.PERFORMANCE_DEGRADATION
        ] and issue.component.startswith('scraper_')
    
    async def remediate(self, issue: HealthIssue, context: Dict[str, Any]) -> RemediationAction:
        action = RemediationAction(
            issue_id=issue.id,
            action_type='restart_scraper',
            parameters={'scraper_id': issue.component}
        )
        
        try:
            scraper_id = issue.component
            docker_client = docker.from_env()
            
            # Find container
            containers = docker_client.containers.list(
                filters={'label': f'scraper_id={scraper_id}'}
            )
            
            if containers:
                container = containers[0]
                logger.info(f"Restarting container for scraper {scraper_id}")
                
                # Restart container
                container.restart(timeout=30)
                
                # Wait for health check
                await asyncio.sleep(10)
                
                # Verify it's running
                container.reload()
                if container.status == 'running':
                    action.status = RemediationStatus.SUCCEEDED
                    action.result = {'container_id': container.id, 'status': 'running'}
                else:
                    action.status = RemediationStatus.FAILED
                    action.result = {'error': f'Container status: {container.status}'}
            else:
                # Start new container
                logger.info(f"Starting new container for scraper {scraper_id}")
                
                container = docker_client.containers.run(
                    'openpolicy/scraper-worker:latest',
                    environment={
                        'SCRAPER_ID': scraper_id,
                        'DATABASE_URL': os.getenv('DATABASE_URL'),
                        'REDIS_URL': os.getenv('REDIS_URL')
                    },
                    labels={'scraper_id': scraper_id},
                    detach=True,
                    restart_policy={'Name': 'unless-stopped'}
                )
                
                action.status = RemediationStatus.SUCCEEDED
                action.result = {'container_id': container.id, 'action': 'started_new'}
                
        except Exception as e:
            logger.error(f"Failed to restart scraper: {e}")
            action.status = RemediationStatus.FAILED
            action.result = {'error': str(e)}
        
        return action
    
    def get_priority(self, issue: HealthIssue) -> int:
        if issue.severity == 'critical':
            return 100
        elif issue.severity == 'high':
            return 80
        else:
            return 50


class DatabaseConnectionStrategy(RemediationStrategy):
    """Strategy for fixing database connection issues"""
    
    async def can_handle(self, issue: HealthIssue) -> bool:
        return issue.type == HealthIssueType.DATABASE_CONNECTION
    
    async def remediate(self, issue: HealthIssue, context: Dict[str, Any]) -> RemediationAction:
        action = RemediationAction(
            issue_id=issue.id,
            action_type='reset_db_connections',
            parameters={}
        )
        
        try:
            # Reset connection pool
            db_url = os.getenv('DATABASE_URL')
            engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=3600)
            
            # Test connection
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                if result.scalar() == 1:
                    action.status = RemediationStatus.SUCCEEDED
                    action.result = {'message': 'Database connection restored'}
                    
                    # Clear connection pool in all workers
                    await self._broadcast_pool_reset()
                else:
                    raise Exception("Database test query failed")
                    
        except Exception as e:
            logger.error(f"Failed to reset database connections: {e}")
            action.status = RemediationStatus.FAILED
            action.result = {'error': str(e)}
        
        return action
    
    async def _broadcast_pool_reset(self):
        """Broadcast connection pool reset to all workers"""
        redis_client = await redis.from_url(os.getenv('REDIS_URL'))
        await redis_client.publish('worker:commands', json.dumps({
            'command': 'reset_db_pool',
            'timestamp': datetime.utcnow().isoformat()
        }))
        await redis_client.close()
    
    def get_priority(self, issue: HealthIssue) -> int:
        return 95  # Database issues are high priority


class QueueManagementStrategy(RemediationStrategy):
    """Strategy for managing queue backup issues"""
    
    async def can_handle(self, issue: HealthIssue) -> bool:
        return issue.type == HealthIssueType.QUEUE_BACKUP
    
    async def remediate(self, issue: HealthIssue, context: Dict[str, Any]) -> RemediationAction:
        action = RemediationAction(
            issue_id=issue.id,
            action_type='manage_queue',
            parameters={}
        )
        
        try:
            redis_client = await redis.from_url(os.getenv('REDIS_URL'))
            
            # Get queue size
            queue_size = await redis_client.llen('scraper:queue')
            
            if queue_size > 1000:
                # Scale up workers
                logger.info(f"Queue size {queue_size}, scaling up workers")
                
                docker_client = docker.from_env()
                
                # Get current worker count
                workers = docker_client.containers.list(
                    filters={'label': 'role=scraper-worker'}
                )
                current_count = len(workers)
                
                # Scale up by 50%
                new_workers = max(2, int(current_count * 0.5))
                
                for i in range(new_workers):
                    container = docker_client.containers.run(
                        'openpolicy/scraper-worker:latest',
                        environment={
                            'DATABASE_URL': os.getenv('DATABASE_URL'),
                            'REDIS_URL': os.getenv('REDIS_URL'),
                            'WORKER_TYPE': 'temporary'
                        },
                        labels={
                            'role': 'scraper-worker',
                            'temporary': 'true',
                            'expires': (datetime.utcnow() + timedelta(hours=1)).isoformat()
                        },
                        detach=True,
                        remove=True
                    )
                
                action.status = RemediationStatus.SUCCEEDED
                action.result = {
                    'action': 'scaled_up',
                    'new_workers': new_workers,
                    'total_workers': current_count + new_workers
                }
                
            else:
                # Queue is manageable
                action.status = RemediationStatus.SUCCEEDED
                action.result = {'message': 'Queue size acceptable', 'size': queue_size}
            
            await redis_client.close()
            
        except Exception as e:
            logger.error(f"Failed to manage queue: {e}")
            action.status = RemediationStatus.FAILED
            action.result = {'error': str(e)}
        
        return action
    
    def get_priority(self, issue: HealthIssue) -> int:
        queue_size = issue.metrics.get('queue_size', 0)
        if queue_size > 5000:
            return 90
        elif queue_size > 1000:
            return 70
        else:
            return 50


class DiskSpaceStrategy(RemediationStrategy):
    """Strategy for managing disk space issues"""
    
    async def can_handle(self, issue: HealthIssue) -> bool:
        return issue.type == HealthIssueType.DISK_SPACE
    
    async def remediate(self, issue: HealthIssue, context: Dict[str, Any]) -> RemediationAction:
        action = RemediationAction(
            issue_id=issue.id,
            action_type='clean_disk_space',
            parameters={}
        )
        
        try:
            # Clean up old logs
            log_dir = '/app/logs'
            cutoff_date = datetime.utcnow() - timedelta(days=7)
            
            cleaned_size = 0
            for root, dirs, files in os.walk(log_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.getmtime(file_path) < cutoff_date.timestamp():
                        size = os.path.getsize(file_path)
                        os.remove(file_path)
                        cleaned_size += size
            
            # Clean up old scraper cache
            cache_dir = '/app/cache'
            if os.path.exists(cache_dir):
                subprocess.run(['find', cache_dir, '-type', 'f', '-mtime', '+3', '-delete'])
            
            # Clean Docker system
            docker_client = docker.from_env()
            docker_client.containers.prune()
            docker_client.images.prune()
            
            # Check new disk usage
            disk_usage = psutil.disk_usage('/')
            
            action.status = RemediationStatus.SUCCEEDED
            action.result = {
                'cleaned_logs': f"{cleaned_size / 1024 / 1024:.2f} MB",
                'disk_usage_percent': disk_usage.percent,
                'free_space_gb': disk_usage.free / 1024 / 1024 / 1024
            }
            
        except Exception as e:
            logger.error(f"Failed to clean disk space: {e}")
            action.status = RemediationStatus.FAILED
            action.result = {'error': str(e)}
        
        return action
    
    def get_priority(self, issue: HealthIssue) -> int:
        disk_usage = issue.metrics.get('disk_usage_percent', 0)
        if disk_usage > 95:
            return 100
        elif disk_usage > 90:
            return 85
        else:
            return 60


class RateLimitStrategy(RemediationStrategy):
    """Strategy for handling rate limit issues"""
    
    async def can_handle(self, issue: HealthIssue) -> bool:
        return issue.type == HealthIssueType.RATE_LIMIT
    
    async def remediate(self, issue: HealthIssue, context: Dict[str, Any]) -> RemediationAction:
        action = RemediationAction(
            issue_id=issue.id,
            action_type='adjust_rate_limits',
            parameters={'scraper_id': issue.component}
        )
        
        try:
            scraper_id = issue.component
            redis_client = await redis.from_url(os.getenv('REDIS_URL'))
            
            # Get current rate limit hits
            rate_limit_key = f"rate_limit:{scraper_id}"
            hits = await redis_client.get(rate_limit_key)
            
            if hits and int(hits) > 10:
                # Implement exponential backoff
                backoff_minutes = min(60, 2 ** (int(hits) - 10))
                
                # Pause scraper temporarily
                await redis_client.setex(
                    f"scraper:paused:{scraper_id}",
                    backoff_minutes * 60,
                    json.dumps({
                        'reason': 'rate_limit',
                        'until': (datetime.utcnow() + timedelta(minutes=backoff_minutes)).isoformat()
                    })
                )
                
                # Adjust scraper rate limit configuration
                await self._update_scraper_rate_limit(scraper_id, backoff_minutes)
                
                action.status = RemediationStatus.SUCCEEDED
                action.result = {
                    'action': 'paused_scraper',
                    'backoff_minutes': backoff_minutes,
                    'rate_limit_hits': int(hits)
                }
            else:
                action.status = RemediationStatus.SUCCEEDED
                action.result = {'message': 'Rate limit acceptable'}
            
            await redis_client.close()
            
        except Exception as e:
            logger.error(f"Failed to handle rate limit: {e}")
            action.status = RemediationStatus.FAILED
            action.result = {'error': str(e)}
        
        return action
    
    async def _update_scraper_rate_limit(self, scraper_id: str, backoff_minutes: int):
        """Update scraper rate limit configuration"""
        # This would update the scraper configuration in the database
        pass
    
    def get_priority(self, issue: HealthIssue) -> int:
        return 70  # Rate limits are medium-high priority


class DataQualityStrategy(RemediationStrategy):
    """Strategy for handling data quality issues"""
    
    async def can_handle(self, issue: HealthIssue) -> bool:
        return issue.type == HealthIssueType.DATA_QUALITY
    
    async def remediate(self, issue: HealthIssue, context: Dict[str, Any]) -> RemediationAction:
        action = RemediationAction(
            issue_id=issue.id,
            action_type='improve_data_quality',
            parameters={'scraper_id': issue.component}
        )
        
        try:
            scraper_id = issue.component
            quality_score = issue.metrics.get('quality_score', 0)
            
            if quality_score < 0.5:
                # Severe quality issues - disable scraper for review
                logger.warning(f"Disabling scraper {scraper_id} due to low quality score: {quality_score}")
                
                # Update scraper status
                engine = create_engine(os.getenv('DATABASE_URL'))
                with engine.connect() as conn:
                    conn.execute(text("""
                        UPDATE scrapers 
                        SET status = 'maintenance',
                            notes = :notes
                        WHERE id = :scraper_id
                    """), {
                        'scraper_id': scraper_id,
                        'notes': f'Auto-disabled due to quality score {quality_score:.2f}'
                    })
                    conn.commit()
                
                # Send notification
                await self._send_quality_alert(scraper_id, quality_score)
                
                action.status = RemediationStatus.REQUIRES_MANUAL
                action.result = {
                    'action': 'disabled_scraper',
                    'reason': 'low_quality_score',
                    'score': quality_score
                }
                
            else:
                # Minor quality issues - adjust validation
                logger.info(f"Adjusting validation for scraper {scraper_id}")
                
                # Enable stricter validation
                redis_client = await redis.from_url(os.getenv('REDIS_URL'))
                await redis_client.hset(
                    f"scraper:config:{scraper_id}",
                    "strict_validation",
                    "true"
                )
                await redis_client.close()
                
                action.status = RemediationStatus.SUCCEEDED
                action.result = {
                    'action': 'enabled_strict_validation',
                    'quality_score': quality_score
                }
                
        except Exception as e:
            logger.error(f"Failed to handle data quality issue: {e}")
            action.status = RemediationStatus.FAILED
            action.result = {'error': str(e)}
        
        return action
    
    async def _send_quality_alert(self, scraper_id: str, quality_score: float):
        """Send alert about data quality issues"""
        # This would integrate with notification system
        pass
    
    def get_priority(self, issue: HealthIssue) -> int:
        quality_score = issue.metrics.get('quality_score', 1.0)
        if quality_score < 0.3:
            return 85
        elif quality_score < 0.6:
            return 65
        else:
            return 40


class AutomatedHealthRemediation:
    """Main automated health remediation system"""
    
    def __init__(self):
        self.strategies: List[RemediationStrategy] = [
            ScraperRestartStrategy(),
            DatabaseConnectionStrategy(),
            QueueManagementStrategy(),
            DiskSpaceStrategy(),
            RateLimitStrategy(),
            DataQualityStrategy()
        ]
        self.active_remediations: Dict[str, RemediationAction] = {}
        self.remediation_history: List[RemediationAction] = []
        self.health_checks = self._setup_health_checks()
        
    def _setup_health_checks(self) -> Dict[str, Any]:
        """Setup health check configurations"""
        return {
            'scraper_failure': {
                'interval': 60,  # seconds
                'threshold': 5,  # consecutive failures
                'severity_mapping': {
                    5: 'medium',
                    10: 'high',
                    20: 'critical'
                }
            },
            'error_rate': {
                'interval': 300,
                'threshold': 0.2,
                'window': 300,  # 5 minutes
                'severity_mapping': {
                    0.2: 'medium',
                    0.5: 'high',
                    0.8: 'critical'
                }
            },
            'disk_space': {
                'interval': 600,
                'threshold': 80,  # percent
                'severity_mapping': {
                    80: 'medium',
                    90: 'high',
                    95: 'critical'
                }
            },
            'queue_size': {
                'interval': 120,
                'threshold': 500,
                'severity_mapping': {
                    500: 'medium',
                    1000: 'high',
                    5000: 'critical'
                }
            }
        }
    
    async def start_monitoring(self):
        """Start the health monitoring and remediation loop"""
        logger.info("Starting automated health remediation system")
        
        # Start health check tasks
        tasks = []
        for check_name, config in self.health_checks.items():
            task = asyncio.create_task(
                self._run_health_check_loop(check_name, config)
            )
            tasks.append(task)
        
        # Start remediation processor
        tasks.append(asyncio.create_task(self._process_remediations()))
        
        # Start cleanup task
        tasks.append(asyncio.create_task(self._cleanup_old_remediations()))
        
        await asyncio.gather(*tasks)
    
    async def _run_health_check_loop(self, check_name: str, config: Dict[str, Any]):
        """Run a specific health check continuously"""
        while True:
            try:
                issues = await self._perform_health_check(check_name, config)
                
                for issue in issues:
                    await self._handle_health_issue(issue)
                
                await asyncio.sleep(config['interval'])
                
            except Exception as e:
                logger.error(f"Error in health check {check_name}: {e}")
                await asyncio.sleep(config['interval'])
    
    async def _perform_health_check(self, check_name: str, config: Dict[str, Any]) -> List[HealthIssue]:
        """Perform a specific health check"""
        issues = []
        
        if check_name == 'scraper_failure':
            issues = await self._check_scraper_failures(config)
        elif check_name == 'error_rate':
            issues = await self._check_error_rate(config)
        elif check_name == 'disk_space':
            issues = await self._check_disk_space(config)
        elif check_name == 'queue_size':
            issues = await self._check_queue_size(config)
        
        return issues
    
    async def _check_scraper_failures(self, config: Dict[str, Any]) -> List[HealthIssue]:
        """Check for scraper failures"""
        issues = []
        
        engine = create_engine(os.getenv('DATABASE_URL'))
        with engine.connect() as conn:
            # Get scrapers with consecutive failures
            result = conn.execute(text("""
                SELECT 
                    s.id,
                    s.name,
                    s.failure_count,
                    s.last_run,
                    s.last_success
                FROM scrapers s
                WHERE s.status = 'active'
                    AND s.failure_count >= :threshold
            """), {'threshold': config['threshold']})
            
            for row in result:
                # Determine severity
                severity = 'low'
                for failures, sev in sorted(config['severity_mapping'].items()):
                    if row.failure_count >= failures:
                        severity = sev
                
                issue = HealthIssue(
                    id=f"scraper_failure_{row.id}_{datetime.utcnow().timestamp()}",
                    type=HealthIssueType.SCRAPER_FAILURE,
                    severity=severity,
                    component=row.id,
                    description=f"Scraper {row.name} has failed {row.failure_count} consecutive times",
                    detected_at=datetime.utcnow(),
                    metrics={
                        'failure_count': row.failure_count,
                        'last_success': row.last_success.isoformat() if row.last_success else None
                    }
                )
                issues.append(issue)
        
        return issues
    
    async def _check_error_rate(self, config: Dict[str, Any]) -> List[HealthIssue]:
        """Check system-wide error rate"""
        issues = []
        
        engine = create_engine(os.getenv('DATABASE_URL'))
        with engine.connect() as conn:
            # Calculate error rate
            result = conn.execute(text("""
                SELECT 
                    COUNT(*) as total_runs,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_runs
                FROM scraper_runs
                WHERE start_time > NOW() - INTERVAL ':window seconds'
            """), {'window': config['window']}).fetchone()
            
            if result.total_runs > 0:
                error_rate = result.failed_runs / result.total_runs
                
                if error_rate >= config['threshold']:
                    # Determine severity
                    severity = 'low'
                    for rate, sev in sorted(config['severity_mapping'].items()):
                        if error_rate >= rate:
                            severity = sev
                    
                    issue = HealthIssue(
                        id=f"high_error_rate_{datetime.utcnow().timestamp()}",
                        type=HealthIssueType.HIGH_ERROR_RATE,
                        severity=severity,
                        component='system',
                        description=f"System error rate is {error_rate:.2%}",
                        detected_at=datetime.utcnow(),
                        metrics={
                            'error_rate': error_rate,
                            'total_runs': result.total_runs,
                            'failed_runs': result.failed_runs
                        }
                    )
                    issues.append(issue)
        
        return issues
    
    async def _check_disk_space(self, config: Dict[str, Any]) -> List[HealthIssue]:
        """Check disk space usage"""
        issues = []
        
        disk_usage = psutil.disk_usage('/')
        
        if disk_usage.percent >= config['threshold']:
            # Determine severity
            severity = 'low'
            for percent, sev in sorted(config['severity_mapping'].items()):
                if disk_usage.percent >= percent:
                    severity = sev
            
            issue = HealthIssue(
                id=f"disk_space_{datetime.utcnow().timestamp()}",
                type=HealthIssueType.DISK_SPACE,
                severity=severity,
                component='system',
                description=f"Disk usage at {disk_usage.percent:.1f}%",
                detected_at=datetime.utcnow(),
                metrics={
                    'disk_usage_percent': disk_usage.percent,
                    'free_space_gb': disk_usage.free / 1024 / 1024 / 1024,
                    'total_space_gb': disk_usage.total / 1024 / 1024 / 1024
                }
            )
            issues.append(issue)
        
        return issues
    
    async def _check_queue_size(self, config: Dict[str, Any]) -> List[HealthIssue]:
        """Check queue size"""
        issues = []
        
        redis_client = await redis.from_url(os.getenv('REDIS_URL'))
        queue_size = await redis_client.llen('scraper:queue')
        await redis_client.close()
        
        if queue_size >= config['threshold']:
            # Determine severity
            severity = 'low'
            for size, sev in sorted(config['severity_mapping'].items()):
                if queue_size >= size:
                    severity = sev
            
            issue = HealthIssue(
                id=f"queue_backup_{datetime.utcnow().timestamp()}",
                type=HealthIssueType.QUEUE_BACKUP,
                severity=severity,
                component='queue',
                description=f"Queue size is {queue_size} items",
                detected_at=datetime.utcnow(),
                metrics={
                    'queue_size': queue_size
                }
            )
            issues.append(issue)
        
        return issues
    
    async def _handle_health_issue(self, issue: HealthIssue):
        """Handle a detected health issue"""
        logger.info(f"Handling health issue: {issue.type.value} - {issue.description}")
        
        # Find appropriate strategy
        for strategy in self.strategies:
            if await strategy.can_handle(issue):
                # Check if already being remediated
                if issue.id not in self.active_remediations:
                    # Create remediation action
                    action = await strategy.remediate(issue, {})
                    action.started_at = datetime.utcnow()
                    
                    # Track active remediation
                    self.active_remediations[issue.id] = action
                    
                    # Log to database
                    await self._log_remediation(issue, action)
                    
                break
    
    async def _process_remediations(self):
        """Process active remediations"""
        while True:
            try:
                # Check active remediations
                completed = []
                
                for issue_id, action in self.active_remediations.items():
                    if action.status in [RemediationStatus.SUCCEEDED, RemediationStatus.FAILED]:
                        action.completed_at = datetime.utcnow()
                        self.remediation_history.append(action)
                        completed.append(issue_id)
                        
                        # Log completion
                        await self._log_remediation_completion(action)
                        
                        # Handle failed remediations
                        if action.status == RemediationStatus.FAILED and action.can_retry():
                            # Retry after delay
                            await asyncio.sleep(30)
                            action.attempts += 1
                            action.status = RemediationStatus.PENDING
                            # Re-add to active
                            self.active_remediations[issue_id] = action
                            completed.remove(issue_id)
                
                # Remove completed remediations
                for issue_id in completed:
                    del self.active_remediations[issue_id]
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error processing remediations: {e}")
                await asyncio.sleep(10)
    
    async def _cleanup_old_remediations(self):
        """Clean up old remediation history"""
        while True:
            try:
                # Keep only last 1000 remediations
                if len(self.remediation_history) > 1000:
                    self.remediation_history = self.remediation_history[-1000:]
                
                # Clean up database entries older than 30 days
                engine = create_engine(os.getenv('DATABASE_URL'))
                with engine.connect() as conn:
                    conn.execute(text("""
                        DELETE FROM remediation_history
                        WHERE completed_at < NOW() - INTERVAL '30 days'
                    """))
                    conn.commit()
                
                await asyncio.sleep(3600)  # Run hourly
                
            except Exception as e:
                logger.error(f"Error cleaning up remediations: {e}")
                await asyncio.sleep(3600)
    
    async def _log_remediation(self, issue: HealthIssue, action: RemediationAction):
        """Log remediation attempt to database"""
        engine = create_engine(os.getenv('DATABASE_URL'))
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO remediation_history 
                (issue_id, issue_type, severity, component, action_type, status, started_at, context)
                VALUES (:issue_id, :issue_type, :severity, :component, :action_type, :status, :started_at, :context)
            """), {
                'issue_id': issue.id,
                'issue_type': issue.type.value,
                'severity': issue.severity,
                'component': issue.component,
                'action_type': action.action_type,
                'status': action.status.value,
                'started_at': action.started_at,
                'context': json.dumps({
                    'issue': issue.to_dict(),
                    'action': action.parameters
                })
            })
            conn.commit()
    
    async def _log_remediation_completion(self, action: RemediationAction):
        """Log remediation completion"""
        engine = create_engine(os.getenv('DATABASE_URL'))
        with engine.connect() as conn:
            conn.execute(text("""
                UPDATE remediation_history
                SET status = :status,
                    completed_at = :completed_at,
                    result = :result
                WHERE issue_id = :issue_id
                    AND started_at = :started_at
            """), {
                'issue_id': action.issue_id,
                'started_at': action.started_at,
                'status': action.status.value,
                'completed_at': action.completed_at,
                'result': json.dumps(action.result)
            })
            conn.commit()
    
    def get_remediation_stats(self) -> Dict[str, Any]:
        """Get remediation statistics"""
        stats = {
            'active_remediations': len(self.active_remediations),
            'total_remediations': len(self.remediation_history),
            'success_rate': 0.0,
            'by_type': defaultdict(int),
            'by_status': defaultdict(int)
        }
        
        if self.remediation_history:
            successful = sum(1 for r in self.remediation_history if r.status == RemediationStatus.SUCCEEDED)
            stats['success_rate'] = successful / len(self.remediation_history)
            
            for remediation in self.remediation_history:
                stats['by_type'][remediation.action_type] += 1
                stats['by_status'][remediation.status.value] += 1
        
        return stats


# Example usage
async def run_health_remediation():
    """Run the automated health remediation system"""
    remediation = AutomatedHealthRemediation()
    await remediation.start_monitoring()


if __name__ == "__main__":
    asyncio.run(run_health_remediation())