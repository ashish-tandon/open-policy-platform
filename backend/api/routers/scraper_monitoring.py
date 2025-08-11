"""
Scraper Monitoring API Endpoints
Provides comprehensive monitoring and control for scraper operations
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
import psutil
import json
import os
from datetime import datetime, timedelta
from pydantic import BaseModel
from ...config.database import db_config

router = APIRouter(prefix="/api/v1/scraper-monitoring", tags=["scraper-monitoring"])

# Data models
class ScraperStatus(BaseModel):
    name: str
    category: str
    status: str
    last_run: Optional[str]
    success_rate: float
    records_collected: int
    error_count: int

class SystemHealth(BaseModel):
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    active_processes: int
    timestamp: str

class DataCollectionStats(BaseModel):
    total_records: int
    records_today: int
    success_rate: float
    active_scrapers: int
    failed_scrapers: int

class ScraperRunRequest(BaseModel):
    category: Optional[str] = None
    max_records: int = 500
    force_run: bool = False

@router.get("/status", response_model=List[ScraperStatus])
async def get_scraper_status():
    """Get comprehensive status of all scrapers"""
    try:
        # Read scraper test reports
        scraper_status = []
        
        # Look for recent test reports
        report_files = [f for f in os.listdir('.') if f.startswith('scraper_test_report_')]
        if report_files:
            latest_report = max(report_files)
            with open(latest_report, 'r') as f:
                report_data = json.load(f)
            
            # Process detailed results
            for scraper_info in report_data.get('detailed_results', []):
                status = ScraperStatus(
                    name=scraper_info.get('name', 'Unknown'),
                    category=scraper_info.get('category', 'Unknown'),
                    status=scraper_info.get('status', 'Unknown'),
                    last_run=scraper_info.get('timestamp'),
                    success_rate=scraper_info.get('success_rate', 0.0),
                    records_collected=scraper_info.get('records_collected', 0),
                    error_count=scraper_info.get('error_count', 0)
                )
                scraper_status.append(status)
        
        return scraper_status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting scraper status: {str(e)}")

@router.get("/health", response_model=SystemHealth)
async def get_system_health():
    """Get system health metrics"""
    try:
        health = SystemHealth(
            cpu_percent=psutil.cpu_percent(),
            memory_percent=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage('/').percent,
            active_processes=len(psutil.pids()),
            timestamp=datetime.now().isoformat()
        )
        return health
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system health: {str(e)}")

@router.get("/stats", response_model=DataCollectionStats)
async def get_data_collection_stats():
    """Get data collection statistics"""
    try:
        # Calculate stats from database
        import subprocess
        
        result = subprocess.run([
            "psql", "-h", db_config.host, "-U", db_config.username, "-d", db_config.database,
            "-c", "SELECT COUNT(*) FROM core_politician;",
            "-t", "-A"
        ], capture_output=True, text=True)
        
        total_records = 0
        if result.returncode == 0 and result.stdout.strip():
            total_records = int(result.stdout.strip())
        
        # Read collection reports for today's stats
        today = datetime.now().strftime("%Y%m%d")
        collection_reports = [f for f in os.listdir('.') if f.startswith(f'collection_report_{today}')]
        
        records_today = 0
        success_rate = 0.0
        active_scrapers = 0
        failed_scrapers = 0
        
        if collection_reports:
            latest_report = max(collection_reports)
            with open(latest_report, 'r') as f:
                report_data = json.load(f)
            
            summary = report_data.get('summary', {})
            records_today = summary.get('total_successes', 0)
            success_rate = summary.get('success_rate', 0.0)
            active_scrapers = summary.get('total_successes', 0)
            failed_scrapers = summary.get('total_failures', 0)
        
        stats = DataCollectionStats(
            total_records=total_records,
            records_today=records_today,
            success_rate=success_rate,
            active_scrapers=active_scrapers,
            failed_scrapers=failed_scrapers
        )
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting data collection stats: {str(e)}")

@router.post("/run")
async def run_scrapers(request: ScraperRunRequest, background_tasks: BackgroundTasks):
    """Run scrapers manually"""
    try:
        # Add scraper run to background tasks
        background_tasks.add_task(run_scraper_background, request)
        
        return {
            "message": "Scraper run initiated",
            "category": request.category or "all",
            "max_records": request.max_records,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initiating scraper run: {str(e)}")

@router.get("/logs")
async def get_scraper_logs(limit: int = 50):
    """Get recent scraper logs"""
    try:
        logs = []
        
        # Read recent log files
        log_files = [
            f for f in os.listdir('.') 
            if f.endswith('.log') and ('scraper' in f or 'collection' in f)
        ]
        
        for log_file in sorted(log_files, reverse=True)[:5]:  # Last 5 log files
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    recent_lines = lines[-limit:]
                    
                    for line in recent_lines:
                        if line.strip():
                            logs.append({
                                "file": log_file,
                                "line": line.strip(),
                                "timestamp": datetime.now().isoformat()
                            })
            except Exception as e:
                continue
        
        return {"logs": logs[-limit:]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting scraper logs: {str(e)}")

@router.get("/failures")
async def get_failure_analysis():
    """Get detailed failure analysis"""
    try:
        failure_analysis = {
            "total_failures": 0,
            "failure_types": {},
            "recent_failures": [],
            "recommendations": []
        }
        
        # Read collection reports for failure analysis
        collection_reports = [f for f in os.listdir('.') if f.startswith('collection_report_')]
        
        if collection_reports:
            latest_report = max(collection_reports)
            with open(latest_report, 'r') as f:
                report_data = json.load(f)
            
            recent_failures = report_data.get('recent_activity', {}).get('failures', [])
            failure_analysis["recent_failures"] = recent_failures
            failure_analysis["total_failures"] = len(recent_failures)
            
            # Analyze failure types
            failure_types = {}
            for failure in recent_failures:
                message = failure.get('message', '')
                if 'classification' in message:
                    failure_types['classification_error'] = failure_types.get('classification_error', 0) + 1
                elif 'SSL' in message:
                    failure_types['ssl_error'] = failure_types.get('ssl_error', 0) + 1
                elif 'timeout' in message:
                    failure_types['timeout'] = failure_types.get('timeout', 0) + 1
                else:
                    failure_types['other'] = failure_types.get('other', 0) + 1
            
            failure_analysis["failure_types"] = failure_types
            
            # Generate recommendations
            recommendations = []
            if failure_types.get('classification_error', 0) > 0:
                recommendations.append("Fix classification errors in CSV scrapers")
            if failure_types.get('ssl_error', 0) > 0:
                recommendations.append("Address SSL certificate issues")
            if failure_types.get('timeout', 0) > 0:
                recommendations.append("Increase timeout values for slow scrapers")
            
            failure_analysis["recommendations"] = recommendations
        
        return failure_analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting failure analysis: {str(e)}")

@router.get("/database/status")
async def get_database_status():
    """Get database status and record counts"""
    try:
        import subprocess
        
        # Get table record counts
        result = subprocess.run([
            "psql", "-h", db_config.host, "-U", db_config.username, "-d", db_config.database,
            "-c", """
            SELECT 
                schemaname, 
                tablename, 
                n_tup_ins as inserts,
                n_tup_upd as updates,
                n_tup_del as deletes
            FROM pg_stat_user_tables 
            WHERE schemaname = 'public' 
            ORDER BY n_tup_ins DESC 
            LIMIT 20;
            """,
            "-t", "-A"
        ], capture_output=True, text=True)
        
        tables = []
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 5:
                        tables.append({
                            "schema": parts[0].strip(),
                            "table": parts[1].strip(),
                            "inserts": int(parts[2].strip()) if parts[2].strip().isdigit() else 0,
                            "updates": int(parts[3].strip()) if parts[3].strip().isdigit() else 0,
                            "deletes": int(parts[4].strip()) if parts[4].strip().isdigit() else 0
                        })
        
        # Get database size
        size_result = subprocess.run([
            "psql", "-h", db_config.host, "-U", db_config.username, "-d", db_config.database,
            "-c", f"SELECT pg_size_pretty(pg_database_size('{db_config.database}'));",
            "-t", "-A"
        ], capture_output=True, text=True)
        
        db_size = "Unknown"
        if size_result.returncode == 0 and size_result.stdout.strip():
            db_size = size_result.stdout.strip()
        
        return {
            "database_size": db_size,
            "tables": tables,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting database status: {str(e)}")

async def run_scraper_background(request: ScraperRunRequest):
    """Background task to run scrapers"""
    try:
        import subprocess
        
        cmd = [
            "python", "scraper_testing_framework.py",
            "--max-sample-records", str(request.max_records),
            "--verbose"
        ]
        
        if request.category:
            cmd.extend(["--category", request.category])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        # Log the result
        log_file = f"manual_scraper_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        with open(log_file, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"Output: {result.stdout}\n")
            f.write(f"Error: {result.stderr}\n")
        
    except Exception as e:
        # Log error
        error_log = f"scraper_run_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        with open(error_log, 'w') as f:
            f.write(f"Error running scrapers: {str(e)}\n")
