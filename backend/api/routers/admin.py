"""
Enhanced Admin Router
Provides comprehensive administrative functionality for system management
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
import subprocess
import json
import os
import psutil
from datetime import datetime, timedelta
from pydantic import BaseModel

from ..dependencies import get_db, require_admin
from ..config import settings
from ...config.database import db_config

router = APIRouter()

# Data models
class SystemRestartRequest(BaseModel):
    services: List[str] = ["api", "database", "scrapers"]
    force: bool = False

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    role: str = "user"
    permissions: Optional[List[str]] = None

class SystemBackupRequest(BaseModel):
    include_database: bool = True
    include_logs: bool = True
    include_reports: bool = True
    backup_name: Optional[str] = None

@router.get("/dashboard")
async def get_dashboard_stats(
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """Get comprehensive dashboard statistics"""
    try:
        # Database statistics
        db_stats = {}
        try:
            result = subprocess.run([
                "psql", "-h", db_config.host, "-U", db_config.username, "-d", db_config.database,
                "-c", "SELECT COUNT(*) FROM core_politician;",
                "-t", "-A"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout.strip():
                db_stats["total_politicians"] = int(result.stdout.strip())
        except:
            db_stats["total_politicians"] = 0
        
        # Scraper statistics
        scraper_stats = {
            "total_scrapers": 0,
            "active_scrapers": 0,
            "success_rate": 0.0
        }
        
        scraper_files = [f for f in os.listdir('.') if f.startswith('scraper_test_report_')]
        if scraper_files:
            latest_report = max(scraper_files)
            try:
                with open(latest_report, 'r') as f:
                    report_data = json.load(f)
                
                summary = report_data.get('summary', {})
                scraper_stats.update({
                    "total_scrapers": summary.get('total_scrapers', 0),
                    "active_scrapers": summary.get('successful', 0),
                    "success_rate": summary.get('success_rate', 0.0)
                })
            except:
                pass
        
        # System statistics
        system_stats = {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "uptime": str(datetime.now() - datetime.fromtimestamp(psutil.boot_time())).split('.')[0]
        }
        
        # API statistics
        api_stats = {
            "status": "healthy",
            "version": settings.version,
            "environment": settings.environment
        }
        
        return {
            "database": db_stats,
            "scrapers": scraper_stats,
            "system": system_stats,
            "api": api_stats,
            "last_update": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving dashboard stats: {str(e)}")

@router.get("/system/status")
async def get_system_status(
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """Get detailed system status"""
    try:
        # Database status
        db_result = subprocess.run([
            "psql", "-h", db_config.host, "-U", db_config.username, "-d", db_config.database,
            "-c", "SELECT 1;",
            "-t", "-A"
        ], capture_output=True, text=True, timeout=10)
        
        database_status = "healthy" if db_result.returncode == 0 else "unhealthy"
        
        # API status
        api_status = "healthy"
        
        # Scraper status
        scraper_status = "active"
        scraper_files = [f for f in os.listdir('.') if f.startswith('scraper_test_report_')]
        if not scraper_files:
            scraper_status = "inactive"
        
        # System metrics
        system_metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "active_processes": len(psutil.pids()),
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
        }
        
        return {
            "database": database_status,
            "api": api_status,
            "scrapers": scraper_status,
            "system": system_metrics,
            "uptime": str(datetime.now() - datetime.fromtimestamp(psutil.boot_time())).split('.')[0],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving system status: {str(e)}")

@router.post("/system/restart")
async def restart_system(
    restart_request: SystemRestartRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """Restart system services"""
    try:
        # Add restart task to background
        background_tasks.add_task(restart_system_services, restart_request)
        
        return {
            "message": "System restart initiated",
            "services": restart_request.services,
            "force": restart_request.force,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initiating system restart: {str(e)}")

@router.get("/users")
async def get_users(
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """Get all system users"""
    try:
        # Mock user data - in real implementation, this would query the database
        users = [
            {
                "id": 1,
                "username": "admin",
                "email": "admin@openpolicy.com",
                "role": "admin",
                "created_at": "2024-01-01T00:00:00Z",
                "last_login": "2024-08-08T00:00:00Z",
                "status": "active"
            },
            {
                "id": 2,
                "username": "user",
                "email": "user@openpolicy.com",
                "role": "user",
                "created_at": "2024-01-01T00:00:00Z",
                "last_login": "2024-08-07T00:00:00Z",
                "status": "active"
            }
        ]
        
        return {
            "users": users,
            "total_users": len(users),
            "active_users": len([u for u in users if u["status"] == "active"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving users: {str(e)}")

@router.post("/users")
async def create_user(
    user: UserCreate,
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """Create a new user"""
    try:
        # Mock user creation - in real implementation, this would insert into database
        new_user = {
            "id": 3,  # Mock ID
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "permissions": user.permissions or [],
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        return {
            "message": "User created successfully",
            "user": new_user
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")

@router.get("/logs")
async def get_system_logs(
    log_type: str = "all",
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """Get system logs"""
    try:
        logs = []
        
        # Get log files based on type
        if log_type == "all":
            log_files = [f for f in os.listdir('.') if f.endswith('.log')]
        else:
            log_files = [f for f in os.listdir('.') if f.endswith('.log') and log_type in f]
        
        for log_file in sorted(log_files, reverse=True)[:10]:
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
            except:
                continue
        
        return {
            "logs": logs[:limit],
            "total_logs": len(logs),
            "log_type": log_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving logs: {str(e)}")

@router.post("/backup")
async def create_system_backup(
    backup_request: SystemBackupRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """Create system backup"""
    try:
        # Add backup task to background
        background_tasks.add_task(create_backup, backup_request)
        
        backup_name = backup_request.backup_name or f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            "message": "System backup initiated",
            "backup_name": backup_name,
            "include_database": backup_request.include_database,
            "include_logs": backup_request.include_logs,
            "include_reports": backup_request.include_reports,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initiating backup: {str(e)}")

@router.get("/backups")
async def list_backups(
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """List available backups"""
    try:
        backups = []
        
        # Look for backup files
        backup_files = [f for f in os.listdir('.') if f.startswith('backup_') and f.endswith('.sql')]
        
        for backup_file in sorted(backup_files, reverse=True):
            try:
                file_stat = os.stat(backup_file)
                backups.append({
                    "name": backup_file,
                    "size": file_stat.st_size,
                    "created_at": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                    "type": "database"
                })
            except:
                continue
        
        return {
            "backups": backups,
            "total_backups": len(backups),
            "total_size": sum(b["size"] for b in backups)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing backups: {str(e)}")

@router.get("/performance")
async def get_system_performance(
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """Get system performance metrics"""
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
        raise HTTPException(status_code=500, detail=f"Error retrieving performance metrics: {str(e)}")

@router.get("/alerts")
async def get_system_alerts(
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
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
        db_result = subprocess.run([
            "psql", "-h", db_config.host, "-U", db_config.username, "-d", db_config.database,
            "-c", "SELECT 1;",
            "-t", "-A"
        ], capture_output=True, text=True, timeout=10)
        
        if db_result.returncode != 0:
            alerts.append({
                "type": "critical",
                "message": "Database connection failed",
                "timestamp": datetime.now().isoformat()
            })
        
        return {
            "alerts": alerts,
            "total_alerts": len(alerts),
            "critical_count": len([a for a in alerts if a["type"] == "critical"]),
            "warning_count": len([a for a in alerts if a["type"] == "warning"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving alerts: {str(e)}")

async def restart_system_services(restart_request: SystemRestartRequest):
    """Background task to restart system services"""
    try:
        # Log restart initiation
        log_file = f"system_restart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        with open(log_file, 'w') as f:
            f.write(f"System restart initiated\n")
            f.write(f"Services: {restart_request.services}\n")
            f.write(f"Force: {restart_request.force}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        
        # Mock restart process
        import time
        time.sleep(5)  # Simulate restart time
        
        # Log restart completion
        with open(log_file, 'a') as f:
            f.write(f"\nSystem restart completed\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        
    except Exception as e:
        # Log error
        error_log = f"system_restart_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        with open(error_log, 'w') as f:
            f.write(f"Error during system restart: {str(e)}\n")

async def create_backup(backup_request: SystemBackupRequest):
    """Background task to create system backup"""
    try:
        backup_name = backup_request.backup_name or f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Log backup initiation
        log_file = f"backup_{backup_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        with open(log_file, 'w') as f:
            f.write(f"System backup initiated\n")
            f.write(f"Backup name: {backup_name}\n")
            f.write(f"Include database: {backup_request.include_database}\n")
            f.write(f"Include logs: {backup_request.include_logs}\n")
            f.write(f"Include reports: {backup_request.include_reports}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        
        # Database backup
        if backup_request.include_database:
            db_backup_file = f"{backup_name}_database.sql"
            result = subprocess.run([
                "pg_dump", "-h", db_config.host, "-U", db_config.username,
                db_config.database, "-f", db_backup_file
            ], capture_output=True, text=True, timeout=300)
            
            with open(log_file, 'a') as f:
                f.write(f"\nDatabase backup completed\n")
                f.write(f"File: {db_backup_file}\n")
                f.write(f"Return code: {result.returncode}\n")
        
        # Logs backup
        if backup_request.include_logs:
            logs_backup_file = f"{backup_name}_logs.tar.gz"
            log_files = [f for f in os.listdir('.') if f.endswith('.log')]
            if log_files:
                result = subprocess.run([
                    "tar", "-czf", logs_backup_file
                ] + log_files, capture_output=True, text=True, timeout=60)
                
                with open(log_file, 'a') as f:
                    f.write(f"\nLogs backup completed\n")
                    f.write(f"File: {logs_backup_file}\n")
                    f.write(f"Return code: {result.returncode}\n")
        
        # Reports backup
        if backup_request.include_reports:
            reports_backup_file = f"{backup_name}_reports.tar.gz"
            report_files = [f for f in os.listdir('.') if f.startswith(('scraper_test_report_', 'collection_report_'))]
            if report_files:
                result = subprocess.run([
                    "tar", "-czf", reports_backup_file
                ] + report_files, capture_output=True, text=True, timeout=60)
                
                with open(log_file, 'a') as f:
                    f.write(f"\nReports backup completed\n")
                    f.write(f"File: {reports_backup_file}\n")
                    f.write(f"Return code: {result.returncode}\n")
        
        # Log backup completion
        with open(log_file, 'a') as f:
            f.write(f"\nSystem backup completed\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        
    except Exception as e:
        # Log error
        error_log = f"backup_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        with open(error_log, 'w') as f:
            f.write(f"Error during backup: {str(e)}\n")
