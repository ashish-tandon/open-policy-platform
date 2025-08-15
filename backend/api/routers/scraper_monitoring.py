"""
Scraper Monitoring API Endpoints
Provides comprehensive monitoring and control for scraper operations
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from typing import List, Dict, Any, Optional
import psutil
import json
import os
from datetime import datetime, timedelta
from pydantic import BaseModel
import logging
from sqlalchemy import text as sql_text

router = APIRouter(prefix="/api/v1/scrapers", tags=["scraper-monitoring"])
logger = logging.getLogger("openpolicy.api.scrapers")

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

class ScraperSummary(BaseModel):
	timestamp: str
	total_scrapers: int
	successful: int
	failed: int
	success_rate: float
	total_records: int

class ScraperRun(BaseModel):
	id: int
	category: str
	start_time: Optional[str]
	end_time: Optional[str]
	status: str
	records_collected: int

@router.get("/summary", response_model=Optional[ScraperSummary])
async def get_latest_summary():
	"""Return the latest scraper summary from scrapers DB if table exists."""
	try:
		try:
			from config.database import scrapers_engine
		except Exception:
			scrapers_engine = None
		if scrapers_engine is None:
			return None
		with scrapers_engine.connect() as conn:
			row = conn.execute(sql_text(
				"""
				SELECT timestamp, total_scrapers, successful, failed, success_rate, total_records
				FROM scraper_results
				ORDER BY id DESC
				LIMIT 1
				"""
			)).fetchone()
			if not row:
				return None
			return ScraperSummary(
				timestamp=str(row[0]),
				total_scrapers=int(row[1] or 0),
				successful=int(row[2] or 0),
				failed=int(row[3] or 0),
				success_rate=float(row[4] or 0.0),
				total_records=int(row[5] or 0),
			)
	except Exception as e:
		# Table may not exist yet; return None without throwing
		logger.info("No scraper summary available: %s", e)
		return None

class ScraperRunRequest(BaseModel):
	category: Optional[str] = None
	max_records: int = 500
	force_run: bool = False

@router.get("/runs", response_model=List[ScraperRun])
async def list_runs(category: Optional[str] = None, status: Optional[str] = None, limit: int = 20):
	"""List recent scraper runs from scrapers DB"""
	try:
		try:
			from config.database import scrapers_engine
		except Exception:
			scrapers_engine = None
		if scrapers_engine is None:
			return []
		query = """
			SELECT id, category, start_time, end_time, status, COALESCE(records_collected, 0)
			FROM scraper_runs
			{where}
			ORDER BY id DESC
			LIMIT :limit
		"""
		clauses = []
		params: Dict[str, Any] = {"limit": int(limit)}
		if category:
			clauses.append("category = :category")
			params["category"] = category
		if status:
			clauses.append("status = :status")
			params["status"] = status
		where_clause = ("WHERE " + " AND ".join(clauses)) if clauses else ""
		with scrapers_engine.connect() as conn:
			rows = conn.execute(sql_text(query.format(where=where_clause)), params).fetchall()
			return [
				ScraperRun(
					id=int(r[0]),
					category=str(r[1]),
					start_time=str(r[2]) if r[2] else None,
					end_time=str(r[3]) if r[3] else None,
					status=str(r[4]),
					records_collected=int(r[5] or 0),
				)
				for r in rows
			]
	except Exception as e:
		logger.warning("Error listing runs: %s", e)
		return []

@router.get("/runs/latest", response_model=Optional[ScraperRun])
async def latest_run(category: Optional[str] = None):
	"""Fetch latest run, optionally filtered by category"""
	try:
		try:
			from config.database import scrapers_engine
		except Exception:
			scrapers_engine = None
		if scrapers_engine is None:
			return None
		query = """
			SELECT id, category, start_time, end_time, status, COALESCE(records_collected, 0)
			FROM scraper_runs
			{where}
			ORDER BY id DESC
			LIMIT 1
		"""
		where_clause = ""
		params: Dict[str, Any] = {}
		if category:
			where_clause = "WHERE category = :category"
			params["category"] = category
		with scrapers_engine.connect() as conn:
			row = conn.execute(sql_text(query.format(where=where_clause)), params).fetchone()
			if not row:
				return None
			return ScraperRun(
				id=int(row[0]),
				category=str(row[1]),
				start_time=str(row[2]) if row[2] else None,
				end_time=str(row[3]) if row[3] else None,
				status=str(row[4]),
				records_collected=int(row[5] or 0),
			)
	except Exception as e:
		logger.warning("Error fetching latest run: %s", e)
		return None

@router.get("/runs/{run_id}", response_model=Optional[ScraperRun])
async def get_run(run_id: int):
	"""Get a specific run by id"""
	try:
		try:
			from config.database import scrapers_engine
		except Exception:
			scrapers_engine = None
		if scrapers_engine is None:
			return None
		with scrapers_engine.connect() as conn:
			row = conn.execute(sql_text(
				"""
				SELECT id, category, start_time, end_time, status, COALESCE(records_collected, 0)
				FROM scraper_runs
				WHERE id = :id
				"""
			), {"id": int(run_id)}).fetchone()
			if not row:
				return None
			return ScraperRun(
				id=int(row[0]),
				category=str(row[1]),
				start_time=str(row[2]) if row[2] else None,
				end_time=str(row[3]) if row[3] else None,
				status=str(row[4]),
				records_collected=int(row[5] or 0),
			)
	except Exception as e:
		logger.warning("Error fetching run %s: %s", run_id, e)
		return None

@router.get("/runs/{run_id}/attempts")
async def list_run_attempts(run_id: int):
	"""List attempts for a run id (if attempts table exists)"""
	try:
		try:
			from config.database import scrapers_engine
		except Exception:
			scrapers_engine = None
		if scrapers_engine is None:
			return {"attempts": []}
		with scrapers_engine.connect() as conn:
			rows = conn.execute(sql_text(
				"""
				SELECT id, scraper_name, attempt_number, started_at, finished_at, status, error_message
				FROM scraper_attempts
				WHERE run_id = :rid
				ORDER BY attempt_number ASC
				"""
			), {"rid": int(run_id)}).fetchall()
			attempts = []
			for r in rows:
				attempts.append({
					"id": int(r[0]),
					"scraper_name": str(r[1]),
					"attempt_number": int(r[2]),
					"started_at": str(r[3]) if r[3] else None,
					"finished_at": str(r[4]) if r[4] else None,
					"status": str(r[5]),
					"error_message": str(r[6]) if r[6] else None,
				})
			return {"attempts": attempts}
	except Exception as e:
		logger.warning("Error listing attempts for run %s: %s", run_id, e)
		return {"attempts": []}

@router.get("/status", response_model=List[ScraperStatus])
async def get_scraper_status(request: Request):
	"""Get comprehensive status of all scrapers"""
	try:
		reports_dir = getattr(request.app.state, "scraper_reports_dir", os.getcwd())
		scraper_status: List[ScraperStatus] = []
		report_files = [
			os.path.join(reports_dir, f) for f in os.listdir(reports_dir)
			if f.startswith('scraper_test_report_')
		]
		if report_files:
			latest_report = max(report_files)
			with open(latest_report, 'r') as f:
				report_data = json.load(f)
			for scraper_info in report_data.get('detailed_results', []):
				try:
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
				except Exception as e:
					logger.warning("Skipping malformed scraper result: %s", e)
		else:
			logger.warning("No scraper reports found in %s", reports_dir)
		return scraper_status
	except Exception as e:
		logger.error("Error getting scraper status: %s", e)
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
		logger.error("Error getting system health: %s", e)
		raise HTTPException(status_code=500, detail=f"Error getting system health: {str(e)}")

@router.get("/stats", response_model=DataCollectionStats)
async def get_data_collection_stats(request: Request):
	"""Get data collection statistics"""
	try:
		try:
			from config.database import engine
		except Exception:
			from ..config import settings  # fallback
			engine = None
		total_records = 0
		if engine is not None:
			try:
				with engine.connect() as conn:
					row = conn.execute(sql_text("SELECT COUNT(*) FROM core_politician;"))
					row = row.fetchone()
					if row and row[0] is not None:
						total_records = int(row[0])
			except Exception as e:
				logger.warning("DB count query failed: %s", e)

		reports_dir = getattr(request.app.state, "scraper_reports_dir", os.getcwd())
		today = datetime.now().strftime("%Y%m%d")
		collection_reports = [
			os.path.join(reports_dir, f) for f in os.listdir(reports_dir)
			if f.startswith(f'collection_report_{today}')
		]
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
		else:
			logger.info("No collection reports for today in %s", reports_dir)

		stats = DataCollectionStats(
			total_records=total_records,
			records_today=records_today,
			success_rate=success_rate,
			active_scrapers=active_scrapers,
			failed_scrapers=failed_scrapers
		)
		return stats
	except Exception as e:
		logger.error("Error getting data collection stats: %s", e)
		raise HTTPException(status_code=500, detail=f"Error getting data collection stats: {str(e)}")

@router.post("/run")
async def run_scrapers(request: Request, background_tasks: BackgroundTasks):
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
async def get_scraper_logs(request: Request, limit: int = 50):
	"""Get recent scraper logs"""
	try:
		logs: list[dict] = []
		logs_dir = getattr(request.app.state, "scraper_logs_dir", os.getcwd())
		log_files = [
			os.path.join(logs_dir, f)
			for f in os.listdir(logs_dir)
			if f.endswith('.log') and ('scraper' in f or 'collection' in f)
		]
		for log_file in sorted(log_files, reverse=True)[:5]:
			try:
				with open(log_file, 'r') as f:
					lines = f.readlines()
				for line in lines[-limit:]:
					if line.strip():
						logs.append({
							"file": os.path.basename(log_file),
							"line": line.strip(),
							"timestamp": datetime.now().isoformat()
						})
			except Exception as e:
				logger.warning("Skipping log file %s: %s", log_file, e)
		return {"logs": logs[-limit:]}
	except Exception as e:
		logger.error("Error reading scraper logs: %s", e)
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
		try:
			from config.database import engine
		except Exception:
			engine = None
		tables = []
		db_size = "Unknown"
		if engine is not None:
			try:
				with engine.connect() as conn:
					rows = conn.execute(sql_text("""
						SELECT 
							table_schema as schemaname, 
							table_name as tablename,
							0 as inserts,
							0 as updates,
							0 as deletes
						FROM information_schema.tables
						WHERE table_schema = 'public'
						ORDER BY table_name ASC
						LIMIT 20;
					""")).fetchall()
					for r in rows:
						tables.append({
							"schema": r[0],
							"table": r[1],
							"inserts": 0,
							"updates": 0,
							"deletes": 0
						})
					size_row = conn.execute(sql_text("SELECT pg_size_pretty(pg_database_size(current_database()));")).fetchone()
					if size_row and size_row[0]:
						db_size = size_row[0]
			except Exception as e:
				logger.warning("DB table/size query failed: %s", e)
		
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
		
		framework_path = os.path.join(os.getcwd(), "OpenPolicyAshBack", "scraper_testing_framework.py")
		if not os.path.exists(framework_path):
			framework_path = "scraper_testing_framework.py"
		cmd = [
			"python", framework_path,
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
