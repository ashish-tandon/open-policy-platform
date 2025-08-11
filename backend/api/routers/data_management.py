"""
Data Management API Endpoints
Provides comprehensive data management, analysis, and export functionality
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import List, Dict, Any, Optional
import subprocess
import json
import csv
import io
from datetime import datetime, timedelta
from pydantic import BaseModel
from ...config.database import db_config

router = APIRouter(prefix="/api/v1/data", tags=["data-management"])

# Data models
class TableInfo(BaseModel):
    table_name: str
    record_count: int
    size_mb: float
    last_updated: Optional[str]

class DataExportRequest(BaseModel):
    table_name: str
    format: str = "json"  # json, csv, sql
    limit: Optional[int] = None
    filters: Optional[Dict[str, Any]] = None

class DataAnalysisResult(BaseModel):
    analysis_type: str
    results: Dict[str, Any]
    timestamp: str

@router.get("/tables", response_model=List[TableInfo])
async def get_table_info():
    """Get information about all tables in the database"""
    try:
        result = subprocess.run([
            "psql", "-h", db_config.host, "-U", db_config.username, "-d", db_config.database,
            "-c", """
            SELECT 
                schemaname,
                tablename,
                n_tup_ins as record_count,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
            FROM pg_stat_user_tables 
            WHERE schemaname = 'public' 
            ORDER BY n_tup_ins DESC;
            """,
            "-t", "-A"
        ], capture_output=True, text=True)
        
        tables = []
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 4:
                        size_str = parts[3].strip()
                        size_mb = 0.0
                        if 'MB' in size_str:
                            size_mb = float(size_str.replace(' MB', ''))
                        elif 'GB' in size_str:
                            size_mb = float(size_str.replace(' GB', '')) * 1024
                        
                        table_info = TableInfo(
                            table_name=parts[1].strip(),
                            record_count=int(parts[2].strip()) if parts[2].strip().isdigit() else 0,
                            size_mb=size_mb,
                            last_updated=datetime.now().isoformat()
                        )
                        tables.append(table_info)
        
        return tables
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting table info: {str(e)}")

@router.get("/tables/{table_name}/records")
async def get_table_records(
    table_name: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Get records from a specific table"""
    try:
        # Validate table name to prevent SQL injection
        valid_tables = [
            'core_politician', 'bills_bill', 'hansards_statement',
            'bills_membervote', 'core_organization', 'core_membership'
        ]
        
        if table_name not in valid_tables:
            raise HTTPException(status_code=400, detail="Invalid table name")
        
        result = subprocess.run([
            "psql", "-h", db_config.host, "-U", db_config.username, "-d", db_config.database,
            "-c", f"SELECT * FROM {table_name} LIMIT {limit} OFFSET {offset};",
            "-t", "-A"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Database error: {result.stderr}")
        
        # Parse the results
        lines = result.stdout.strip().split('\n')
        records = []
        
        for line in lines:
            if line.strip():
                # Simple CSV-like parsing
                fields = line.split('|')
                records.append(fields)
        
        return {
            "table_name": table_name,
            "records": records,
            "total_returned": len(records),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting table records: {str(e)}")

@router.post("/export")
async def export_data(request: DataExportRequest, background_tasks: BackgroundTasks):
    """Export data from a specific table"""
    try:
        # Validate table name
        valid_tables = [
            'core_politician', 'bills_bill', 'hansards_statement',
            'bills_membervote', 'core_organization', 'core_membership'
        ]
        
        if request.table_name not in valid_tables:
            raise HTTPException(status_code=400, detail="Invalid table name")
        
        # Add export task to background
        background_tasks.add_task(export_data_background, request)
        
        return {
            "message": "Export initiated",
            "table_name": request.table_name,
            "format": request.format,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initiating export: {str(e)}")

@router.get("/analysis/politicians")
async def analyze_politicians():
    """Analyze politician data"""
    try:
        # Get politician statistics
        result = subprocess.run([
            "psql", "-h", db_config.host, "-U", db_config.username, "-d", db_config.database,
            "-c", """
            SELECT 
                COUNT(*) as total_politicians,
                COUNT(DISTINCT party_name) as total_parties,
                COUNT(DISTINCT district) as total_districts
            FROM core_politician;
            """,
            "-t", "-A"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Database error: {result.stderr}")
        
        # Parse results
        line = result.stdout.strip()
        if '|' in line:
            parts = line.split('|')
            analysis = {
                "total_politicians": int(parts[0].strip()) if parts[0].strip().isdigit() else 0,
                "total_parties": int(parts[1].strip()) if parts[1].strip().isdigit() else 0,
                "total_districts": int(parts[2].strip()) if parts[2].strip().isdigit() else 0
            }
        else:
            analysis = {"error": "Could not parse results"}
        
        return DataAnalysisResult(
            analysis_type="politicians",
            results=analysis,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing politicians: {str(e)}")

@router.get("/analysis/bills")
async def analyze_bills():
    """Analyze bill data"""
    try:
        # Get bill statistics
        result = subprocess.run([
            "psql", "-h", db_config.host, "-U", db_config.username, "-d", db_config.database,
            "-c", """
            SELECT 
                COUNT(*) as total_bills,
                COUNT(DISTINCT session) as total_sessions,
                COUNT(DISTINCT classification) as total_classifications
            FROM bills_bill;
            """,
            "-t", "-A"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Database error: {result.stderr}")
        
        # Parse results
        line = result.stdout.strip()
        if '|' in line:
            parts = line.split('|')
            analysis = {
                "total_bills": int(parts[0].strip()) if parts[0].strip().isdigit() else 0,
                "total_sessions": int(parts[1].strip()) if parts[1].strip().isdigit() else 0,
                "total_classifications": int(parts[2].strip()) if parts[2].strip().isdigit() else 0
            }
        else:
            analysis = {"error": "Could not parse results"}
        
        return DataAnalysisResult(
            analysis_type="bills",
            results=analysis,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing bills: {str(e)}")

@router.get("/analysis/hansards")
async def analyze_hansards():
    """Analyze hansard (parliamentary debate) data"""
    try:
        # Get hansard statistics
        result = subprocess.run([
            "psql", "-h", "localhost", "-U", "ashishtandon", "-d", "openpolicy",
            "-c", """
            SELECT 
                COUNT(*) as total_statements,
                COUNT(DISTINCT speaker_name) as total_speakers,
                COUNT(DISTINCT date) as total_dates
            FROM hansards_statement;
            """,
            "-t", "-A"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Database error: {result.stderr}")
        
        # Parse results
        line = result.stdout.strip()
        if '|' in line:
            parts = line.split('|')
            analysis = {
                "total_statements": int(parts[0].strip()) if parts[0].strip().isdigit() else 0,
                "total_speakers": int(parts[1].strip()) if parts[1].strip().isdigit() else 0,
                "total_dates": int(parts[2].strip()) if parts[2].strip().isdigit() else 0
            }
        else:
            analysis = {"error": "Could not parse results"}
        
        return DataAnalysisResult(
            analysis_type="hansards",
            results=analysis,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing hansards: {str(e)}")

@router.get("/search")
async def search_data(
    query: str = Query(..., min_length=2),
    table_name: str = Query("core_politician"),
    limit: int = Query(50, ge=1, le=200)
):
    """Search data across tables"""
    try:
        # Validate table name
        valid_tables = [
            'core_politician', 'bills_bill', 'hansards_statement',
            'bills_membervote', 'core_organization', 'core_membership'
        ]
        
        if table_name not in valid_tables:
            raise HTTPException(status_code=400, detail="Invalid table name")
        
        # Build search query based on table
        if table_name == 'core_politician':
            search_query = f"""
            SELECT * FROM {table_name} 
            WHERE name ILIKE '%{query}%' 
            OR party_name ILIKE '%{query}%' 
            OR district ILIKE '%{query}%'
            LIMIT {limit};
            """
        elif table_name == 'bills_bill':
            search_query = f"""
            SELECT * FROM {table_name} 
            WHERE title ILIKE '%{query}%' 
            OR classification ILIKE '%{query}%'
            LIMIT {limit};
            """
        else:
            search_query = f"""
            SELECT * FROM {table_name} 
            WHERE text ILIKE '%{query}%'
            LIMIT {limit};
            """
        
        result = subprocess.run([
            "psql", "-h", "localhost", "-U", "ashishtandon", "-d", "openpolicy",
            "-c", search_query,
            "-t", "-A"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Database error: {result.stderr}")
        
        # Parse results
        lines = result.stdout.strip().split('\n')
        records = []
        
        for line in lines:
            if line.strip():
                fields = line.split('|')
                records.append(fields)
        
        return {
            "query": query,
            "table_name": table_name,
            "results": records,
            "total_found": len(records),
            "limit": limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching data: {str(e)}")

@router.get("/database/size")
async def get_database_size():
    """Get database size information"""
    try:
        result = subprocess.run([
            "psql", "-h", "localhost", "-U", "ashishtandon", "-d", "openpolicy",
            "-c", """
            SELECT 
                pg_size_pretty(pg_database_size('openpolicy')) as total_size,
                pg_size_pretty(pg_total_relation_size('core_politician')) as politicians_size,
                pg_size_pretty(pg_total_relation_size('bills_bill')) as bills_size,
                pg_size_pretty(pg_total_relation_size('hansards_statement')) as hansards_size;
            """,
            "-t", "-A"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Database error: {result.stderr}")
        
        line = result.stdout.strip()
        if '|' in line:
            parts = line.split('|')
            size_info = {
                "total_size": parts[0].strip(),
                "politicians_size": parts[1].strip(),
                "bills_size": parts[2].strip(),
                "hansards_size": parts[3].strip()
            }
        else:
            size_info = {"error": "Could not parse results"}
        
        return size_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting database size: {str(e)}")

async def export_data_background(request: DataExportRequest):
    """Background task to export data"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_file = f"export_{request.table_name}_{timestamp}"
        
        if request.format == "csv":
            export_file += ".csv"
            cmd = [
                "psql", "-h", "localhost", "-U", "ashishtandon", "-d", "openpolicy",
                "-c", f"\\COPY {request.table_name} TO '{export_file}' CSV HEADER;"
            ]
        elif request.format == "json":
            export_file += ".json"
            cmd = [
                "psql", "-h", "localhost", "-U", "ashishtandon", "-d", "openpolicy",
                "-c", f"\\COPY (SELECT row_to_json(t) FROM {request.table_name} t) TO '{export_file}';"
            ]
        else:  # sql
            export_file += ".sql"
            cmd = [
                "pg_dump", "-h", "localhost", "-U", "ashishtandon", 
                "-t", request.table_name, "openpolicy", "-f", export_file
            ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Log the export result
        log_file = f"export_log_{timestamp}.log"
        with open(log_file, 'w') as f:
            f.write(f"Export: {request.table_name}\n")
            f.write(f"Format: {request.format}\n")
            f.write(f"File: {export_file}\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"Output: {result.stdout}\n")
            f.write(f"Error: {result.stderr}\n")
        
    except Exception as e:
        # Log error
        error_log = f"export_error_{timestamp}.log"
        with open(error_log, 'w') as f:
            f.write(f"Error exporting {request.table_name}: {str(e)}\n")
