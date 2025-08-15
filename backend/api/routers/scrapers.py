"""
Enhanced Scrapers Router
Provides comprehensive scraper management, execution, and monitoring functionality
"""

from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, Query, Request
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import subprocess
import json
import os
from datetime import datetime, timedelta
from pydantic import BaseModel
from sqlalchemy import text as sql_text

from ..dependencies import get_db, require_admin
from ..config import settings

router = APIRouter()

# Data models
class ScraperExecution(BaseModel):
    scraper_id: str
    category: Optional[str] = None
    max_records: int = 500
    force_run: bool = False

class ScraperStatus(BaseModel):
    scraper_id: str
    name: str
    category: str
    status: str
    last_run: Optional[str]
    success_rate: float
    records_collected: int
    error_count: int

@router.get("/")
async def get_scrapers(request: Request, db: Session = Depends(get_db)):
    """Get comprehensive list of available scrapers with status"""
    try:
        # Read scraper inventory
        scraper_inventory = []
        inventory_file = "SCRAPER_INVENTORY.md"
        
        if os.path.exists(inventory_file):
            with open(inventory_file, 'r') as f:
                content = f.read()
                
            # Parse scraper categories
            categories = {
                "Provincial": [],
                "Municipal": [],
                "Parliamentary": [],
                "Civic": [],
                "Update": []
            }
            
            current_category = None
            for line in content.split('\n'):
                if line.startswith('### '):
                    current_category = line.replace('### ', '').strip()
                elif line.startswith('- ') and current_category:
                    scraper_name = line.replace('- ', '').strip()
                    if scraper_name and current_category in categories:
                        categories[current_category].append({
                            "name": scraper_name,
                            "category": current_category,
                            "status": "available"
                        })
        
        # Flatten categories
        all_scrapers = []
        for category, scrapers in categories.items():
            all_scrapers.extend(scrapers)
        
        # Get recent status from reports
        reports_dir = getattr(request.app.state, "scraper_reports_dir", os.getcwd())
        scraper_files = [f for f in os.listdir(reports_dir) if f.startswith('scraper_test_report_')]
        if scraper_files:
            latest_report = max(scraper_files)
            try:
                with open(os.path.join(reports_dir, latest_report), 'r') as f:
                    report_data = json.load(f)
                
                # Update status from report
                for scraper in all_scrapers:
                    for result in report_data.get('detailed_results', []):
                        if result.get('name') == scraper['name']:
                            scraper['status'] = result.get('status', 'unknown')
                            scraper['last_run'] = result.get('timestamp')
                            scraper['success_rate'] = result.get('success_rate', 0.0)
                            scraper['records_collected'] = result.get('records_collected', 0)
                            scraper['error_count'] = result.get('error_count', 0)
                            break
            except:
                pass
        
        return {
            "scrapers": all_scrapers,
            "total_scrapers": len(all_scrapers),
            "categories": {cat: len(scrapers) for cat, scrapers in categories.items()},
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving scrapers: {str(e)}")

@router.get("/categories")
async def get_scraper_categories(request: Request, db: Session = Depends(get_db)):
    """Get scraper categories with statistics"""
    try:
        categories = {
            "Provincial": {"count": 0, "active": 0, "success_rate": 0.0},
            "Municipal": {"count": 0, "active": 0, "success_rate": 0.0},
            "Parliamentary": {"count": 0, "active": 0, "success_rate": 0.0},
            "Civic": {"count": 0, "active": 0, "success_rate": 0.0},
            "Update": {"count": 0, "active": 0, "success_rate": 0.0}
        }
        
        # Read from inventory
        inventory_file = "SCRAPER_INVENTORY.md"
        if os.path.exists(inventory_file):
            with open(inventory_file, 'r') as f:
                content = f.read()
                
            current_category = None
            for line in content.split('\n'):
                if line.startswith('### '):
                    current_category = line.replace('### ', '').strip()
                elif line.startswith('- ') and current_category and current_category in categories:
                    categories[current_category]["count"] += 1
        
        # Get status from reports
        reports_dir = getattr(request.app.state, "scraper_reports_dir", os.getcwd())
        scraper_files = [f for f in os.listdir(reports_dir) if f.startswith('scraper_test_report_')]
        if scraper_files:
            latest_report = max(scraper_files)
            try:
                with open(os.path.join(reports_dir, latest_report), 'r') as f:
                    report_data = json.load(f)
                
                # Calculate category statistics
                category_stats = {}
                for result in report_data.get('detailed_results', []):
                    category = result.get('category', 'Unknown')
                    if category not in category_stats:
                        category_stats[category] = {"active": 0, "successful": 0, "total": 0}
                    
                    category_stats[category]["total"] += 1
                    if result.get('status') == 'success':
                        category_stats[category]["successful"] += 1
                        category_stats[category]["active"] += 1
                
                # Update categories
                for category, stats in category_stats.items():
                    if category in categories:
                        categories[category]["active"] = stats["active"]
                        if stats["total"] > 0:
                            categories[category]["success_rate"] = round((stats["successful"] / stats["total"]) * 100, 2)
            except:
                pass
        
        return {
            "categories": categories,
            "total_scrapers": sum(cat["count"] for cat in categories.values()),
            "active_scrapers": sum(cat["active"] for cat in categories.values()),
            "overall_success_rate": round(sum(cat["success_rate"] for cat in categories.values()) / len(categories), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving categories: {str(e)}")

@router.post("/{scraper_id}/run")
async def run_scraper(
    scraper_id: str,
    execution: ScraperExecution,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """Run a specific scraper with detailed execution parameters"""
    try:
        # Add scraper execution to background tasks
        background_tasks.add_task(run_scraper_task, execution)
        
        return {
            "message": f"Scraper {scraper_id} execution initiated",
            "scraper_id": scraper_id,
            "execution_params": execution.dict(),
            "task_id": f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initiating scraper execution: {str(e)}")

@router.get("/{scraper_id}/status")
async def get_scraper_status(scraper_id: str, request: Request, db: Session = Depends(get_db)):
    """Get detailed status of a specific scraper"""
    try:
        # Find scraper in reports
        reports_dir = getattr(request.app.state, "scraper_reports_dir", os.getcwd())
        scraper_files = [f for f in os.listdir(reports_dir) if f.startswith('scraper_test_report_')]
        
        scraper_status = {
            "scraper_id": scraper_id,
            "name": scraper_id,
            "category": "Unknown",
            "status": "unknown",
            "last_run": None,
            "success_rate": 0.0,
            "records_collected": 0,
            "error_count": 0,
            "execution_history": []
        }
        
        if scraper_files:
            latest_report = max(scraper_files)
            try:
                with open(os.path.join(reports_dir, latest_report), 'r') as f:
                    report_data = json.load(f)
                
                # Find scraper in results
                for result in report_data.get('detailed_results', []):
                    if result.get('name') == scraper_id or result.get('id') == scraper_id:
                        scraper_status.update({
                            "name": result.get('name', scraper_id),
                            "category": result.get('category', 'Unknown'),
                            "status": result.get('status', 'unknown'),
                            "last_run": result.get('timestamp'),
                            "success_rate": result.get('success_rate', 0.0),
                            "records_collected": result.get('records_collected', 0),
                            "error_count": result.get('error_count', 0)
                        })
                        break
            except:
                pass
        
        # Get execution history from logs
        logs_dir = getattr(request.app.state, "scraper_logs_dir", os.getcwd())
        log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log') and 'scraper' in f]
        for log_file in sorted(log_files, reverse=True)[:5]:
            try:
                with open(os.path.join(logs_dir, log_file), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if scraper_id.lower() in line.lower():
                            scraper_status["execution_history"].append({
                                "log_file": log_file,
                                "message": line.strip(),
                                "timestamp": datetime.now().isoformat()
                            })
            except:
                continue
        
        return scraper_status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving scraper status: {str(e)}")

@router.get("/{scraper_id}/logs")
async def get_scraper_logs(
    scraper_id: str, 
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """Get logs for a specific scraper"""
    try:
        logs = []
        
        # Search for scraper in log files
        logs_dir = getattr(request.app.state, "scraper_logs_dir", os.getcwd())
        log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log') and 'scraper' in f]
        
        for log_file in sorted(log_files, reverse=True)[:10]:
            try:
                with open(os.path.join(logs_dir, log_file), 'r') as f:
                    lines = f.readlines()
                    
                    for line in lines:
                        if scraper_id.lower() in line.lower():
                            logs.append({
                                "log_file": log_file,
                                "line": line.strip(),
                                "timestamp": datetime.now().isoformat()
                            })
                            
                            if len(logs) >= limit:
                                break
                    
                    if len(logs) >= limit:
                        break
            except:
                continue
        
        return {
            "scraper_id": scraper_id,
            "logs": logs[:limit],
            "total_logs": len(logs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving scraper logs: {str(e)}")

@router.post("/run/category/{category}")
async def run_scrapers_by_category(
    category: str,
    execution: ScraperExecution,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """Run all scrapers in a specific category"""
    try:
        # Find scrapers in category
        scrapers_in_category = []
        inventory_file = "SCRAPER_INVENTORY.md"
        
        if os.path.exists(inventory_file):
            with open(inventory_file, 'r') as f:
                content = f.read()
                
            current_cat = None
            for line in content.split('\n'):
                if line.startswith('### '):
                    current_cat = line.replace('### ', '').strip()
                elif line.startswith('- ') and current_cat == category:
                    scraper_name = line.replace('- ', '').strip()
                    if scraper_name:
                        scrapers_in_category.append(scraper_name)
        
        # Add execution tasks
        for scraper in scrapers_in_category:
            execution.scraper_id = scraper
            background_tasks.add_task(run_scraper_task, execution)
        
        return {
            "message": f"Executing {len(scrapers_in_category)} scrapers in category {category}",
            "category": category,
            "scrapers": scrapers_in_category,
            "execution_params": execution.dict(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing category scrapers: {str(e)}")

@router.post("/run/record/{category}")
async def record_category_run(category: str):
    """Record a category run in the scrapers DB (best-effort). Returns run_id."""
    try:
        try:
            from config.database import scrapers_engine
        except Exception:
            scrapers_engine = None
        run_id = None
        if scrapers_engine is not None:
            with scrapers_engine.begin() as conn:
                row = conn.execute(sql_text(
                    """
                    INSERT INTO scraper_runs (category, status) VALUES (:cat, :st) RETURNING id
                    """
                ), {"cat": category, "st": "started"}).fetchone()
                if row:
                    run_id = int(row[0])
        return {"category": category, "run_id": run_id}
    except Exception as e:
        return {"category": category, "run_id": None, "error": str(e)}

@router.post("/run/dev/{category}")
async def run_category_dev(category: str, current_user = Depends(require_admin)):
    """Trigger the dev category runner: records run, executes scanner, updates run."""
    try:
        cmd = [
            "python",
            "scripts/run_category.py",
            "--category",
            category,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return {
            "category": category,
            "returncode": result.returncode,
            "stdout": result.stdout[-500:],
            "stderr": result.stderr[-500:],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running dev category: {e}")

@router.post("/run/full/{category}")
async def run_category_full(category: str, retries: int = 2, max_records: int = 10, current_user = Depends(require_admin)):
    """Trigger the full category runner with retries/backoff and attempt tracking."""
    try:
        cmd = [
            "python",
            "scripts/run_full_category.py",
            "--category",
            category,
            "--retries",
            str(int(retries)),
            "--max-records",
            str(int(max_records)),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        return {
            "category": category,
            "returncode": result.returncode,
            "stdout": result.stdout[-1000:],
            "stderr": result.stderr[-1000:],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running full category: {e}")

@router.post("/queue/full/{category}")
async def queue_category_full(category: str, retries: int = 2, max_records: int = 10, current_user = Depends(require_admin)):
    """Queue the full category runner as a Celery task (requires workers)."""
    try:
        try:
            from ..celery_app import celery_app
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Celery unavailable: {e}")
        task = celery_app.send_task(
            "scrapers.run_category",
            args=[category, int(retries), int(max_records)],
        )
        return {"task_id": task.id, "category": category}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error queueing full category: {e}")

@router.get("/performance")
async def get_scraper_performance(db: Session = Depends(get_db)):
    """Get overall scraper performance metrics"""
    try:
        performance = {
            "total_scrapers": 0,
            "active_scrapers": 0,
            "success_rate": 0.0,
            "total_records_collected": 0,
            "total_errors": 0,
            "category_performance": {},
            "recent_activity": []
        }
        
        # Read from reports
        scraper_files = [f for f in os.listdir('.') if f.startswith('scraper_test_report_')]
        if scraper_files:
            latest_report = max(scraper_files)
            try:
                with open(latest_report, 'r') as f:
                    report_data = json.load(f)
                
                summary = report_data.get('summary', {})
                performance.update({
                    "total_scrapers": summary.get('total_scrapers', 0),
                    "active_scrapers": summary.get('successful', 0),
                    "success_rate": summary.get('success_rate', 0.0),
                    "total_records_collected": summary.get('total_records_collected', 0),
                    "total_errors": summary.get('failed', 0)
                })
                
                # Category performance
                category_stats = {}
                for result in report_data.get('detailed_results', []):
                    category = result.get('category', 'Unknown')
                    if category not in category_stats:
                        category_stats[category] = {"total": 0, "successful": 0, "records": 0}
                    
                    category_stats[category]["total"] += 1
                    if result.get('status') == 'success':
                        category_stats[category]["successful"] += 1
                    category_stats[category]["records"] += result.get('records_collected', 0)
                
                for category, stats in category_stats.items():
                    performance["category_performance"][category] = {
                        "total": stats["total"],
                        "successful": stats["successful"],
                        "success_rate": round((stats["successful"] / stats["total"]) * 100, 2) if stats["total"] > 0 else 0,
                        "records_collected": stats["records"]
                    }
            except:
                pass
        
        return performance
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving performance metrics: {str(e)}")

async def run_scraper_task(execution: ScraperExecution):
    """Background task to run scraper"""
    try:
        import subprocess
        
        framework_path = os.path.join(os.getcwd(), "OpenPolicyAshBack", "scraper_testing_framework.py")
        if not os.path.exists(framework_path):
            framework_path = "scraper_testing_framework.py"
        cmd = [
            "python", framework_path,
            "--max-sample-records", str(execution.max_records),
            "--verbose"
        ]
        
        if execution.category:
            cmd.extend(["--category", execution.category])
        
        # Log execution start
        log_file = f"scraper_execution_{execution.scraper_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        with open(log_file, 'w') as f:
            f.write(f"Starting scraper execution: {execution.scraper_id}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Parameters: {execution.dict()}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        # Log execution result
        with open(log_file, 'a') as f:
            f.write(f"\nExecution completed\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"Output: {result.stdout}\n")
            f.write(f"Error: {result.stderr}\n")
        
    except Exception as e:
        # Log error
        error_log = f"scraper_execution_error_{execution.scraper_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        with open(error_log, 'w') as f:
            f.write(f"Error executing scraper {execution.scraper_id}: {str(e)}\n")
