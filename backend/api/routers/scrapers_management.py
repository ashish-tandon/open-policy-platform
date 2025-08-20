"""
Scraper Management API Router - 40by6 Implementation
API endpoints for managing 1700+ scrapers
"""

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import json

from ..dependencies import get_db
from backend.mcp.scraper_management_system import (
    MCPScraperManagementSystem, ScraperCategory, ScraperPlatform,
    ScraperStatus, ScraperMetadata
)

router = APIRouter(prefix="/api/v1/scrapers", tags=["scraper-management"])

# Global system instance (in production, use dependency injection)
scraper_system = MCPScraperManagementSystem()


class ScraperFilter(BaseModel):
    """Filter criteria for scrapers"""
    category: Optional[ScraperCategory] = None
    platform: Optional[ScraperPlatform] = None
    status: Optional[ScraperStatus] = None
    jurisdiction_type: Optional[str] = None
    jurisdiction_code: Optional[str] = None
    tags: Optional[List[str]] = None


class ScraperUpdate(BaseModel):
    """Update scraper configuration"""
    status: Optional[ScraperStatus] = None
    schedule: Optional[str] = None
    priority: Optional[int] = Field(None, ge=1, le=10)
    config: Optional[Dict[str, Any]] = None


class ScraperExecutionRequest(BaseModel):
    """Request to execute scrapers"""
    scraper_ids: Optional[List[str]] = None
    category: Optional[ScraperCategory] = None
    force: bool = False


class ScraperHealthCheck(BaseModel):
    """Health check configuration"""
    check_connectivity: bool = True
    check_authentication: bool = True
    check_data_quality: bool = True
    sample_size: int = 10


@router.get("/")
async def list_scrapers(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    filter: ScraperFilter = Depends()
) -> Dict[str, Any]:
    """List all scrapers with optional filtering"""
    
    all_scrapers = list(scraper_system.registry.scrapers.values())
    
    # Apply filters
    filtered = all_scrapers
    if filter.category:
        filtered = [s for s in filtered if s.category == filter.category]
    if filter.platform:
        filtered = [s for s in filtered if s.platform == filter.platform]
    if filter.status:
        filtered = [s for s in filtered if s.status == filter.status]
    if filter.jurisdiction_type:
        filtered = [s for s in filtered if s.jurisdiction["type"] == filter.jurisdiction_type]
    if filter.jurisdiction_code:
        filtered = [s for s in filtered if s.jurisdiction["code"] == filter.jurisdiction_code]
    if filter.tags:
        filtered = [s for s in filtered if any(tag in s.tags for tag in filter.tags)]
    
    # Pagination
    total = len(filtered)
    scrapers = filtered[skip:skip + limit]
    
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "scrapers": [s.__dict__ for s in scrapers]
    }


@router.get("/stats")
async def get_scraper_statistics() -> Dict[str, Any]:
    """Get scraper statistics and overview"""
    
    stats = {
        "total_scrapers": len(scraper_system.registry.scrapers),
        "by_category": {},
        "by_platform": {},
        "by_status": {},
        "by_jurisdiction": {
            "federal": 0,
            "provincial": 0,
            "municipal": 0
        },
        "health_metrics": {
            "success_rate_24h": 0.0,
            "average_runtime": 0.0,
            "failed_scrapers": 0,
            "stale_scrapers": 0
        }
    }
    
    # Calculate statistics
    for scraper in scraper_system.registry.scrapers.values():
        # By category
        cat = scraper.category.value
        stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1
        
        # By platform
        plat = scraper.platform.value
        stats["by_platform"][plat] = stats["by_platform"].get(plat, 0) + 1
        
        # By status
        status = scraper.status.value
        stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
        
        # By jurisdiction
        jur_type = scraper.jurisdiction["type"]
        if jur_type in stats["by_jurisdiction"]:
            stats["by_jurisdiction"][jur_type] += 1
        
        # Health metrics
        if scraper.status == ScraperStatus.FAILED:
            stats["health_metrics"]["failed_scrapers"] += 1
        
        if scraper.last_run and (datetime.utcnow() - scraper.last_run) > timedelta(days=7):
            stats["health_metrics"]["stale_scrapers"] += 1
    
    return stats


@router.get("/registry")
async def export_scraper_registry(format: str = "json") -> Dict[str, Any]:
    """Export the complete scraper registry"""
    
    registry_path = scraper_system.registry.export_registry(format)
    
    return {
        "status": "exported",
        "path": registry_path,
        "total_scrapers": len(scraper_system.registry.scrapers),
        "export_time": datetime.utcnow().isoformat()
    }


@router.get("/{scraper_id}")
async def get_scraper_details(scraper_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific scraper"""
    
    if scraper_id not in scraper_system.registry.scrapers:
        raise HTTPException(status_code=404, detail="Scraper not found")
    
    scraper = scraper_system.registry.scrapers[scraper_id]
    
    # Get recent runs
    recent_runs = []  # Would fetch from database
    
    return {
        "scraper": scraper.__dict__,
        "recent_runs": recent_runs,
        "health_status": _calculate_health_status(scraper)
    }


@router.patch("/{scraper_id}")
async def update_scraper(
    scraper_id: str,
    update: ScraperUpdate,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Update scraper configuration"""
    
    if scraper_id not in scraper_system.registry.scrapers:
        raise HTTPException(status_code=404, detail="Scraper not found")
    
    scraper = scraper_system.registry.scrapers[scraper_id]
    
    # Apply updates
    if update.status is not None:
        scraper.status = update.status
    if update.schedule is not None:
        scraper.schedule = update.schedule
    if update.priority is not None:
        scraper.priority = update.priority
    if update.config is not None:
        scraper.config.update(update.config)
    
    scraper.updated_at = datetime.utcnow()
    
    # Save to database
    # db.execute(...)
    
    return {"status": "updated", "scraper": scraper.__dict__}


@router.post("/execute")
async def execute_scrapers(
    request: ScraperExecutionRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Execute scrapers on demand"""
    
    scraper_ids = []
    
    if request.scraper_ids:
        scraper_ids = request.scraper_ids
    elif request.category:
        # Get all scrapers in category
        scrapers = scraper_system.registry.get_scrapers_by_category(request.category)
        scraper_ids = [s.id for s in scrapers]
    else:
        raise HTTPException(status_code=400, detail="Must specify scraper_ids or category")
    
    # Filter by status unless forced
    if not request.force:
        scraper_ids = [
            sid for sid in scraper_ids
            if scraper_system.registry.scrapers[sid].status == ScraperStatus.ACTIVE
        ]
    
    # Schedule execution
    background_tasks.add_task(
        scraper_system.orchestrator.schedule_scrapers,
        scraper_ids
    )
    
    return {
        "status": "scheduled",
        "scheduled_count": len(scraper_ids),
        "scraper_ids": scraper_ids
    }


@router.get("/runs/active")
async def get_active_runs() -> Dict[str, Any]:
    """Get currently running scrapers"""
    
    running = scraper_system.orchestrator.get_running_scrapers()
    
    return {
        "active_count": len(running),
        "runs": [
            {
                "run_id": run.run_id,
                "scraper_id": run.scraper_id,
                "scraper_name": scraper_system.registry.scrapers.get(run.scraper_id, {}).name,
                "start_time": run.start_time.isoformat(),
                "duration": (datetime.utcnow() - run.start_time).total_seconds(),
                "status": run.status
            }
            for run in running
        ]
    }


@router.post("/runs/{run_id}/stop")
async def stop_scraper_run(run_id: str) -> Dict[str, Any]:
    """Stop a running scraper"""
    
    success = await scraper_system.orchestrator.stop_scraper(run_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Run not found or already stopped")
    
    return {"status": "stopped", "run_id": run_id}


@router.post("/health-check")
async def run_health_check(
    config: ScraperHealthCheck,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Run health check on all scrapers"""
    
    job_id = f"health_check_{datetime.utcnow().timestamp()}"
    
    background_tasks.add_task(
        _run_comprehensive_health_check,
        job_id,
        config
    )
    
    return {
        "status": "started",
        "job_id": job_id,
        "estimated_duration": "10-30 minutes"
    }


@router.get("/monitoring/dashboard")
async def get_monitoring_dashboard() -> Dict[str, Any]:
    """Get monitoring dashboard data"""
    
    monitor = scraper_system.monitor
    await monitor.update_metrics()
    
    return {
        "overview": monitor.get_health_report(),
        "performance": {
            "response_times": _get_response_time_histogram(),
            "success_rates": _get_success_rate_timeline(),
            "data_volume": _get_data_volume_metrics()
        },
        "alerts": monitor._check_alerts(),
        "recommendations": monitor._generate_recommendations()
    }


@router.post("/discover")
async def discover_new_scrapers(
    background_tasks: BackgroundTasks,
    paths: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Discover new scrapers in the filesystem"""
    
    background_tasks.add_task(
        scraper_system.registry.discover_scrapers
    )
    
    return {
        "status": "discovery_started",
        "message": "Scraper discovery running in background"
    }


@router.get("/jurisdictions")
async def list_jurisdictions() -> Dict[str, Any]:
    """List all jurisdictions with scraper counts"""
    
    jurisdictions = {}
    
    for scraper in scraper_system.registry.scrapers.values():
        jur_key = f"{scraper.jurisdiction['type']}:{scraper.jurisdiction['code']}"
        if jur_key not in jurisdictions:
            jurisdictions[jur_key] = {
                "type": scraper.jurisdiction["type"],
                "code": scraper.jurisdiction["code"],
                "name": scraper.jurisdiction["name"],
                "scraper_count": 0,
                "active_scrapers": 0
            }
        
        jurisdictions[jur_key]["scraper_count"] += 1
        if scraper.status == ScraperStatus.ACTIVE:
            jurisdictions[jur_key]["active_scrapers"] += 1
    
    return {
        "total_jurisdictions": len(jurisdictions),
        "jurisdictions": list(jurisdictions.values())
    }


@router.get("/platforms/{platform}/scrapers")
async def get_platform_scrapers(
    platform: ScraperPlatform,
    status: Optional[ScraperStatus] = None
) -> Dict[str, Any]:
    """Get all scrapers for a specific platform"""
    
    scrapers = scraper_system.registry.get_scrapers_by_platform(platform)
    
    if status:
        scrapers = [s for s in scrapers if s.status == status]
    
    return {
        "platform": platform.value,
        "total": len(scrapers),
        "scrapers": [s.__dict__ for s in scrapers]
    }


# Helper functions
def _calculate_health_status(scraper: ScraperMetadata) -> Dict[str, Any]:
    """Calculate health status for a scraper"""
    health = {
        "status": "unknown",
        "score": 0.0,
        "issues": []
    }
    
    if scraper.status == ScraperStatus.FAILED:
        health["status"] = "critical"
        health["issues"].append("Scraper in failed state")
    elif scraper.failure_count > 3:
        health["status"] = "warning"
        health["issues"].append(f"High failure count: {scraper.failure_count}")
    elif scraper.last_run and (datetime.utcnow() - scraper.last_run) > timedelta(days=7):
        health["status"] = "warning"
        health["issues"].append("Scraper hasn't run in over 7 days")
    else:
        health["status"] = "healthy"
        health["score"] = 1.0
    
    return health


async def _run_comprehensive_health_check(job_id: str, config: ScraperHealthCheck):
    """Run comprehensive health check on all scrapers"""
    # Implementation of health check logic
    pass


def _get_response_time_histogram() -> List[Dict[str, Any]]:
    """Get response time histogram data"""
    # Implementation
    return []


def _get_success_rate_timeline() -> List[Dict[str, Any]]:
    """Get success rate timeline data"""
    # Implementation
    return []


def _get_data_volume_metrics() -> Dict[str, Any]:
    """Get data volume metrics"""
    # Implementation
    return {
        "total_records_24h": 0,
        "by_type": {},
        "by_jurisdiction": {}
    }