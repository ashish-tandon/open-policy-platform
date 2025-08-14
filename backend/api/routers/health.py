"""
Enhanced Health Check Router
Provides comprehensive health checks, system diagnostics, and monitoring functionality
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
import subprocess
import psutil
import json
import os
from datetime import datetime, timedelta
from pydantic import BaseModel

from ..dependencies import get_db
from ..config import settings

router = APIRouter()

# Data models
class HealthStatus(BaseModel):
    status: str
    service: str
    version: str
    environment: str
    timestamp: str
    uptime: str

class DetailedHealthStatus(BaseModel):
    status: str
    service: str
    version: str
    environment: str
    database: str
    timestamp: str
    uptime: str
    system_metrics: Dict[str, Any]

class SystemDiagnostics(BaseModel):
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    active_processes: int
    load_average: List[float]
    uptime: str

class DatabaseHealth(BaseModel):
    status: str
    connectivity: Optional[str] = None
    database_size: Optional[str] = None
    table_count: Optional[int] = None
    politician_records: Optional[int] = None
    timestamp: Optional[str] = None
    error: Optional[str] = None

class ScraperHealth(BaseModel):
    status: str
    total_scrapers: int
    active_scrapers: int
    success_rate: float
    last_run: Optional[str] = None
    report_file: Optional[str] = None
    timestamp: str
    message: Optional[str] = None

class ApiHealth(BaseModel):
    status: str
    service: str
    version: str
    environment: str
    uptime: str
    endpoints: Dict[str, str]
    timestamp: str

class ComprehensiveHealth(BaseModel):
    status: str
    components: Dict[str, Any]
    summary: Dict[str, int]
    timestamp: str

class Metrics(BaseModel):
    system: Dict[str, Any]
    database: Dict[str, Any]
    scrapers: Dict[str, Any]
    network: Dict[str, Any]
    timestamp: str

@router.get("/health", response_model=HealthStatus)
async def health_check() -> Dict[str, Any]:
    """Basic health check"""
    try:
        # Get system uptime
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = str(datetime.now() - boot_time).split('.')[0]
        
        return {
            "status": "healthy",
            "service": "Open Policy Platform API",
            "version": settings.version,
            "environment": settings.environment,
            "timestamp": datetime.now().isoformat(),
            "uptime": uptime
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "Open Policy Platform API",
            "version": settings.version,
            "environment": settings.environment,
            "timestamp": datetime.now().isoformat(),
            "uptime": "0:00:00"
        }

@router.get("/health/detailed", response_model=DetailedHealthStatus)
async def detailed_health_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Detailed health check with database connectivity and system metrics"""
    try:
        # Test database connection
        db_status = "healthy"
        try:
            from sqlalchemy import text as sql_text
            from ...config.database import engine
            with engine.connect() as conn:
                conn.execute(sql_text("SELECT 1"))
        except Exception as e:
            db_status = f"unhealthy: {str(e)}"
        
        # Get system metrics
        system_metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "active_processes": len(psutil.pids())
        }
        
        # Get system uptime
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = str(datetime.now() - boot_time).split('.')[0]
        
        # Determine overall status
        overall_status = "healthy"
        if db_status != "healthy" or system_metrics["cpu_percent"] > 90 or system_metrics["memory_percent"] > 90:
            overall_status = "unhealthy"
        elif system_metrics["cpu_percent"] > 80 or system_metrics["memory_percent"] > 80:
            overall_status = "warning"
        
        return {
            "status": overall_status,
            "service": "Open Policy Platform API",
            "version": settings.version,
            "environment": settings.environment,
            "database": db_status,
            "timestamp": datetime.now().isoformat(),
            "uptime": uptime,
            "system_metrics": system_metrics
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "Open Policy Platform API",
            "version": settings.version,
            "environment": settings.environment,
            "database": f"unhealthy: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "uptime": "0:00:00",
            "system_metrics": {}
        }

@router.get("/health/database", response_model=DatabaseHealth)
async def database_health_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Database-specific health check"""
    try:
        # Test basic connectivity
        try:
            from sqlalchemy import text as sql_text
            from ...config.database import engine
            with engine.connect() as conn:
                conn.execute(sql_text("SELECT 1"))
        except Exception as e:
            return {
                "status": "unhealthy",
                "connectivity": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        
        # Get database size and counts using SQL
        db_size = "Unknown"
        table_count = 0
        politician_count = 0
        try:
            with engine.connect() as conn:
                size_row = conn.execute(sql_text("SELECT pg_size_pretty(pg_database_size(current_database()));")).fetchone()
                if size_row and size_row[0]:
                    db_size = size_row[0]
                table_row = conn.execute(sql_text("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")).fetchone()
                if table_row and table_row[0] is not None:
                    table_count = int(table_row[0])
                try:
                    pol_row = conn.execute(sql_text("SELECT COUNT(*) FROM core_politician;")).fetchone()
                    if pol_row and pol_row[0] is not None:
                        politician_count = int(pol_row[0])
                except Exception:
                    pass
        except Exception as e:
            return {
                "status": "unhealthy",
                "connectivity": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "status": "healthy",
            "connectivity": "successful",
            "database_size": db_size,
            "table_count": table_count,
            "politician_records": politician_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "connectivity": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/health/scrapers", response_model=ScraperHealth)
async def scraper_health_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Scraper-specific health check"""
    try:
        # Check for scraper reports
        scraper_files = [f for f in os.listdir('.') if f.startswith('scraper_test_report_')]
        
        if not scraper_files:
            return {
                "status": "warning",
                "message": "No scraper reports found",
                "total_scrapers": 0,
                "active_scrapers": 0,
                "success_rate": 0.0,
                "last_run": None,
                "timestamp": datetime.now().isoformat()
            }
        
        # Get latest report
        latest_report = max(scraper_files)
        try:
            with open(latest_report, 'r') as f:
                report_data = json.load(f)
            
            summary = report_data.get('summary', {})
            total_scrapers = summary.get('total_scrapers', 0)
            active_scrapers = summary.get('successful', 0)
            success_rate = summary.get('success_rate', 0.0)
            last_run = report_data.get('timestamp')
            
            # Determine status
            status = "healthy"
            if success_rate < 50:
                status = "unhealthy"
            elif success_rate < 70:
                status = "warning"
            
            return {
                "status": status,
                "total_scrapers": total_scrapers,
                "active_scrapers": active_scrapers,
                "success_rate": success_rate,
                "last_run": last_run,
                "report_file": latest_report,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "warning",
                "message": f"Error reading scraper report: {str(e)}",
                "report_file": latest_report,
                "total_scrapers": 0,
                "active_scrapers": 0,
                "success_rate": 0.0,
                "last_run": None,
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "total_scrapers": 0,
            "active_scrapers": 0,
            "success_rate": 0.0,
            "timestamp": datetime.now().isoformat()
        }

@router.get("/health/system", response_model=SystemDiagnostics)
async def system_health_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """System-specific health check"""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # Network I/O
        network_io = psutil.net_io_counters()
        
        # Active processes
        active_processes = len(psutil.pids())
        
        # Load average (Unix-like systems)
        load_average = []
        try:
            load_avg = os.getloadavg()
            load_average = list(load_avg)
        except:
            load_average = [0.0, 0.0, 0.0]
        
        # System uptime
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = str(datetime.now() - boot_time).split('.')[0]
        
        # Determine status
        status = "healthy"
        if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
            status = "unhealthy"
        elif cpu_percent > 80 or memory_percent > 80 or disk_percent > 80:
            status = "warning"
        
        return {
            "status": status,
            "cpu_usage": cpu_percent,
            "memory_usage": memory_percent,
            "disk_usage": disk_percent,
            "network_io": {
                "bytes_sent": network_io.bytes_sent,
                "bytes_recv": network_io.bytes_recv,
                "packets_sent": network_io.packets_sent,
                "packets_recv": network_io.packets_recv
            },
            "active_processes": active_processes,
            "load_average": load_average,
            "uptime": uptime
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "cpu_usage": 0,
            "memory_usage": 0,
            "disk_usage": 0,
            "network_io": {},
            "active_processes": 0,
            "load_average": [0, 0, 0],
            "uptime": "0:00:00"
        }

@router.get("/health/api", response_model=ApiHealth)
async def api_health_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """API-specific health check"""
    try:
        # Check API version
        api_version = settings.version
        
        # Check environment
        environment = settings.environment
        
        # Check if API is responding
        api_status = "healthy"
        
        # Get API uptime (simulated)
        api_start_time = datetime.now() - timedelta(hours=1)  # Simulated start time
        api_uptime = str(datetime.now() - api_start_time).split('.')[0]
        
        return {
            "status": api_status,
            "service": "Open Policy Platform API",
            "version": api_version,
            "environment": environment,
            "uptime": api_uptime,
            "endpoints": {
                "health": "/health",
                "detailed_health": "/health/detailed",
                "database_health": "/health/database",
                "scraper_health": "/health/scrapers",
                "system_health": "/health/system"
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "Open Policy Platform API",
            "version": settings.version,
            "environment": settings.environment,
            "uptime": "0:00:00",
            "endpoints": {},
            "timestamp": datetime.now().isoformat()
        }

@router.get("/health/comprehensive", response_model=ComprehensiveHealth)
async def comprehensive_health_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Comprehensive health check covering all components"""
    try:
        # Get all health checks
        api_health = await api_health_check(db)
        database_health = await database_health_check(db)
        scraper_health = await scraper_health_check(db)
        system_health = await system_health_check(db)
        
        # Determine overall status
        all_statuses = [
            api_health.get("status"),
            database_health.get("status"),
            scraper_health.get("status"),
            system_health.get("status")
        ]
        
        overall_status = "healthy"
        if "unhealthy" in all_statuses:
            overall_status = "unhealthy"
        elif "warning" in all_statuses:
            overall_status = "warning"
        
        # Compile comprehensive report
        comprehensive_report = {
            "status": overall_status,
            "components": {
                "api": api_health,
                "database": database_health,
                "scrapers": scraper_health,
                "system": system_health
            },
            "summary": {
                "total_components": 4,
                "healthy_components": all_statuses.count("healthy"),
                "warning_components": all_statuses.count("warning"),
                "unhealthy_components": all_statuses.count("unhealthy")
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return comprehensive_report
    except Exception as e:
        return {
            "status": "unhealthy",
            "components": {},
            "summary": {"total_components": 0, "healthy_components": 0, "warning_components": 0, "unhealthy_components": 0},
            "timestamp": datetime.now().isoformat()
        }

@router.get("/health/metrics", response_model=Metrics)
async def health_metrics(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get health metrics for monitoring"""
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        # Database metrics
        db_connected = False
        politician_count = 0
        try:
            from sqlalchemy import text as sql_text
            from ...config.database import engine
            with engine.connect() as conn:
                conn.execute(sql_text("SELECT 1"))
                db_connected = True
                try:
                    pol_row = conn.execute(sql_text("SELECT COUNT(*) FROM core_politician;")); pol_row = pol_row.fetchone()
                    if pol_row and pol_row[0] is not None:
                        politician_count = int(pol_row[0])
                except Exception:
                    pass
        except Exception:
            pass
        
        # Scraper metrics
        scraper_success_rate = 0.0
        scraper_files = [f for f in os.listdir('.') if f.startswith('scraper_test_report_')]
        if scraper_files:
            try:
                latest_report = max(scraper_files)
                with open(latest_report, 'r') as f:
                    report_data = json.load(f)
                
                summary = report_data.get('summary', {})
                scraper_success_rate = summary.get('success_rate', 0.0)
            except:
                pass
        
        # Network metrics
        network_io = psutil.net_io_counters()
        
        metrics = {
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "active_processes": len(psutil.pids())
            },
            "database": {
                "connected": db_connected,
                "politician_records": politician_count
            },
            "scrapers": {
                "success_rate": scraper_success_rate
            },
            "network": {
                "bytes_sent": network_io.bytes_sent,
                "bytes_recv": network_io.bytes_recv
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return metrics
    except Exception as e:
        return {
            "system": {},
            "database": {},
            "scrapers": {},
            "network": {},
            "timestamp": datetime.now().isoformat()
        }
