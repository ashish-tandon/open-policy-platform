"""
Dashboard API Endpoints
Provides comprehensive dashboard functionality for the OpenPolicy platform
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import psutil
import json
import os
from datetime import datetime, timedelta
from pydantic import BaseModel
from ..config import settings
from ..dependencies import get_db
from ...config.database import db_config

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])

# Data models
class DashboardOverview(BaseModel):
    system_status: str
    database_status: str
    scraper_status: str
    api_status: str
    total_records: int
    active_scrapers: int
    success_rate: float
    last_update: str

class SystemMetrics(BaseModel):
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    active_processes: int
    uptime: str

class ScraperMetrics(BaseModel):
    total_scrapers: int
    active_scrapers: int
    failed_scrapers: int
    success_rate: float
    records_collected_today: int
    last_run: str

class DatabaseMetrics(BaseModel):
    total_size: str
    total_tables: int
    total_records: int
    largest_table: str
    last_backup: str

@router.get("/overview", response_model=DashboardOverview)
async def get_dashboard_overview():
    """Get comprehensive dashboard overview"""
    try:
        # System status
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        system_status = "healthy"
        if cpu_percent > 90 or memory_percent > 90:
            system_status = "warning"
        elif cpu_percent > 80 or memory_percent > 80:
            system_status = "attention"
        
        # Database status
        import subprocess
        db_user = db_config.username
        db_result = subprocess.run([
            "psql", "-h", db_config.host, "-U", db_user, "-d", db_config.database,
            "-c", "SELECT 1;",
            "-t", "-A"
        ], capture_output=True, text=True, timeout=10)
        
        database_status = "healthy" if db_result.returncode == 0 else "unhealthy"
        
        # Scraper status
        scraper_status = "active"
        scraper_files = [f for f in os.listdir('.') if f.startswith('scraper_test_report_')]
        if not scraper_files:
            scraper_status = "inactive"
        
        # API status
        api_status = "healthy"
        
        # Get total records
        total_records = 0
        if database_status == "healthy":
            count_result = subprocess.run([
                "psql", "-h", db_config.host, "-U", db_user, "-d", db_config.database,
                "-c", "SELECT COUNT(*) FROM core_politician;",
                "-t", "-A"
            ], capture_output=True, text=True, timeout=10)
            
            if count_result.returncode == 0 and count_result.stdout.strip():
                total_records = int(count_result.stdout.strip())
        
        # Get scraper metrics
        active_scrapers = 0
        success_rate = 0.0
        
        if scraper_files:
            latest_report = max(scraper_files)
            try:
                with open(latest_report, 'r') as f:
                    report_data = json.load(f)
                
                summary = report_data.get('summary', {})
                active_scrapers = summary.get('successful', 0)
                success_rate = summary.get('success_rate', 0.0)
            except:
                pass
        
        overview = DashboardOverview(
            system_status=system_status,
            database_status=database_status,
            scraper_status=scraper_status,
            api_status=api_status,
            total_records=total_records,
            active_scrapers=active_scrapers,
            success_rate=success_rate,
            last_update=datetime.now().isoformat()
        )
        
        return overview
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dashboard overview: {str(e)}")

@router.get("/system", response_model=SystemMetrics)
async def get_system_metrics():
    """Get detailed system metrics"""
    try:
        # CPU and memory
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        
        # Network I/O
        network_io = psutil.net_io_counters()._asdict()
        
        # Active processes
        active_processes = len(psutil.pids())
        
        # Uptime
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = str(datetime.now() - boot_time).split('.')[0]
        
        metrics = SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=network_io,
            active_processes=active_processes,
            uptime=uptime
        )
        
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system metrics: {str(e)}")

@router.get("/scrapers", response_model=ScraperMetrics)
async def get_scraper_metrics():
    """Get scraper performance metrics"""
    try:
        # Find latest scraper report
        scraper_files = [f for f in os.listdir('.') if f.startswith('scraper_test_report_')]
        
        total_scrapers = 0
        active_scrapers = 0
        failed_scrapers = 0
        success_rate = 0.0
        records_collected_today = 0
        last_run = datetime.now().isoformat()
        
        if scraper_files:
            latest_report = max(scraper_files)
            try:
                with open(latest_report, 'r') as f:
                    report_data = json.load(f)
                
                summary = report_data.get('summary', {})
                total_scrapers = summary.get('total_scrapers', 0)
                active_scrapers = summary.get('successful', 0)
                failed_scrapers = summary.get('failed', 0)
                success_rate = summary.get('success_rate', 0.0)
                records_collected_today = summary.get('total_records_collected', 0)
                last_run = report_data.get('timestamp', last_run)
            except:
                pass
        
        metrics = ScraperMetrics(
            total_scrapers=total_scrapers,
            active_scrapers=active_scrapers,
            failed_scrapers=failed_scrapers,
            success_rate=success_rate,
            records_collected_today=records_collected_today,
            last_run=last_run
        )
        
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting scraper metrics: {str(e)}")

@router.get("/database", response_model=DatabaseMetrics)
async def get_database_metrics():
    """Get database performance metrics"""
    try:
        import subprocess
        
        # Database size
        size_result = subprocess.run([
            "psql", "-h", db_config.host, "-U", db_config.username, "-d", db_config.database,
            "-c", f"SELECT pg_size_pretty(pg_database_size('{db_config.database}'));",
            "-t", "-A"
        ], capture_output=True, text=True, timeout=10)
        
        total_size = "Unknown"
        if size_result.returncode == 0 and size_result.stdout.strip():
            total_size = size_result.stdout.strip()
        
        # Table count
        table_result = subprocess.run([
            "psql", "-h", db_config.host, "-U", db_config.username, "-d", db_config.database,
            "-c", "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';",
            "-t", "-A"
        ], capture_output=True, text=True, timeout=10)
        
        total_tables = 0
        if table_result.returncode == 0 and table_result.stdout.strip():
            total_tables = int(table_result.stdout.strip())
        
        # Total records
        records_result = subprocess.run([
            "psql", "-h", db_config.host, "-U", db_config.username, "-d", db_config.database,
            "-c", "SELECT COUNT(*) FROM core_politician;",
            "-t", "-A"
        ], capture_output=True, text=True, timeout=10)
        
        total_records = 0
        if records_result.returncode == 0 and records_result.stdout.strip():
            total_records = int(records_result.stdout.strip())
        
        # Largest table
        largest_result = subprocess.run([
            "psql", "-h", db_config.host, "-U", db_config.username, "-d", db_config.database,
            "-c", """
            SELECT tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
            FROM pg_stat_user_tables 
            WHERE schemaname = 'public' 
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC 
            LIMIT 1;
            """,
            "-t", "-A"
        ], capture_output=True, text=True, timeout=10)
        
        largest_table = "Unknown"
        if largest_result.returncode == 0 and largest_result.stdout.strip():
            line = largest_result.stdout.strip()
            if '|' in line:
                parts = line.split('|')
                largest_table = f"{parts[0].strip()} ({parts[1].strip()})"
        
        # Last backup (simulated)
        last_backup = "Never"
        
        metrics = DatabaseMetrics(
            total_size=total_size,
            total_tables=total_tables,
            total_records=total_records,
            largest_table=largest_table,
            last_backup=last_backup
        )
        
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting database metrics: {str(e)}")

@router.get("/alerts")
async def get_system_alerts():
    """Get system alerts and warnings"""
    try:
        alerts = []
        
        # CPU alert
        cpu_percent = psutil.cpu_percent()
        if cpu_percent > 90:
            alerts.append({
                "type": "critical",
                "message": f"High CPU usage: {cpu_percent}%",
                "timestamp": datetime.now().isoformat()
            })
        elif cpu_percent > 80:
            alerts.append({
                "type": "warning",
                "message": f"Elevated CPU usage: {cpu_percent}%",
                "timestamp": datetime.now().isoformat()
            })
        
        # Memory alert
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 90:
            alerts.append({
                "type": "critical",
                "message": f"High memory usage: {memory_percent}%",
                "timestamp": datetime.now().isoformat()
            })
        elif memory_percent > 80:
            alerts.append({
                "type": "warning",
                "message": f"Elevated memory usage: {memory_percent}%",
                "timestamp": datetime.now().isoformat()
            })
        
        # Disk alert
        disk_usage = psutil.disk_usage('/').percent
        if disk_usage > 90:
            alerts.append({
                "type": "critical",
                "message": f"High disk usage: {disk_usage}%",
                "timestamp": datetime.now().isoformat()
            })
        elif disk_usage > 80:
            alerts.append({
                "type": "warning",
                "message": f"Elevated disk usage: {disk_usage}%",
                "timestamp": datetime.now().isoformat()
            })
        
        # Database alert
        import subprocess
        db_result = subprocess.run([
            "psql", "-h", "localhost", "-U", "ashishtandon", "-d", "openpolicy",
            "-c", "SELECT 1;",
            "-t", "-A"
        ], capture_output=True, text=True, timeout=10)
        
        if db_result.returncode != 0:
            alerts.append({
                "type": "critical",
                "message": "Database connection failed",
                "timestamp": datetime.now().isoformat()
            })
        
        # Scraper alert
        scraper_files = [f for f in os.listdir('.') if f.startswith('scraper_test_report_')]
        if scraper_files:
            latest_report = max(scraper_files)
            try:
                with open(latest_report, 'r') as f:
                    report_data = json.load(f)
                
                summary = report_data.get('summary', {})
                success_rate = summary.get('success_rate', 0.0)
                
                if success_rate < 50:
                    alerts.append({
                        "type": "critical",
                        "message": f"Low scraper success rate: {success_rate}%",
                        "timestamp": datetime.now().isoformat()
                    })
                elif success_rate < 70:
                    alerts.append({
                        "type": "warning",
                        "message": f"Moderate scraper success rate: {success_rate}%",
                        "timestamp": datetime.now().isoformat()
                    })
            except:
                pass
        
        return {
            "alerts": alerts,
            "total_alerts": len(alerts),
            "critical_count": len([a for a in alerts if a["type"] == "critical"]),
            "warning_count": len([a for a in alerts if a["type"] == "warning"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system alerts: {str(e)}")

@router.get("/recent-activity")
async def get_recent_activity(limit: int = 10):
    """Get recent system activity"""
    try:
        activities = []
        
        # Check recent log files
        log_files = [
            f for f in os.listdir('.') 
            if f.endswith('.log') and ('scraper' in f or 'collection' in f or 'api' in f)
        ]
        
        for log_file in sorted(log_files, reverse=True)[:5]:
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    recent_lines = lines[-limit:]
                    
                    for line in recent_lines:
                        if line.strip():
                            activities.append({
                                "type": "log",
                                "source": log_file,
                                "message": line.strip(),
                                "timestamp": datetime.now().isoformat()
                            })
            except:
                continue
        
        # Check recent reports
        report_files = [
            f for f in os.listdir('.') 
            if f.startswith(('scraper_test_report_', 'collection_report_'))
        ]
        
        for report_file in sorted(report_files, reverse=True)[:3]:
            try:
                with open(report_file, 'r') as f:
                    report_data = json.load(f)
                
                activities.append({
                    "type": "report",
                    "source": report_file,
                    "message": f"Generated {report_data.get('summary', {}).get('total_scrapers', 0)} scraper results",
                    "timestamp": report_data.get('timestamp', datetime.now().isoformat())
                })
            except:
                continue
        
        # Sort by timestamp
        activities.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "activities": activities[:limit],
            "total_activities": len(activities)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recent activity: {str(e)}")

@router.get("/performance")
async def get_performance_metrics():
    """Get performance metrics and trends"""
    try:
        # System performance
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        
        # Network performance
        network_io = psutil.net_io_counters()
        
        # Process performance
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                if proc.info['cpu_percent'] > 0 or proc.info['memory_percent'] > 0:
                    processes.append({
                        "pid": proc.info['pid'],
                        "name": proc.info['name'],
                        "cpu_percent": proc.info['cpu_percent'],
                        "memory_percent": proc.info['memory_percent']
                    })
            except:
                continue
        
        # Sort by CPU usage
        processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
        
        performance = {
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_usage": disk_usage
            },
            "network": {
                "bytes_sent": network_io.bytes_sent,
                "bytes_recv": network_io.bytes_recv,
                "packets_sent": network_io.packets_sent,
                "packets_recv": network_io.packets_recv
            },
            "top_processes": processes[:10],
            "timestamp": datetime.now().isoformat()
        }
        
        return performance
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting performance metrics: {str(e)}")
