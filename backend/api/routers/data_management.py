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
from sqlalchemy import text as sql_text
import os

router = APIRouter(prefix="/api/v1/data", tags=["data-management"])

# Data models
class TableInfo(BaseModel):
    table_name: str
    record_count: int
    size_mb: float
    last_updated: Optional[str]

class ColumnInfo(BaseModel):
    table_name: str
    column_name: str
    data_type: str
    is_nullable: bool

class DataExportRequest(BaseModel):
    table_name: str
    format: str = "json"  # json, csv, sql
    limit: Optional[int] = None
    filters: Optional[Dict[str, Any]] = None

class DataAnalysisResult(BaseModel):
    analysis_type: str
    results: Dict[str, Any]
    timestamp: str

@router.get("/schema", response_model=Dict[str, List[ColumnInfo]])
async def get_schema():
    """Introspect current database schema (public schema) via SQLAlchemy."""
    try:
        try:
            from ...config.database import engine
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DB engine unavailable: {e}")
        schema: Dict[str, List[ColumnInfo]] = {}
        with engine.connect() as conn:
            rows = conn.execute(sql_text(
                """
                SELECT table_name, column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position
                """
            )).fetchall()
            for r in rows:
                t = r[0]
                if t not in schema:
                    schema[t] = []
                schema[t].append(ColumnInfo(
                    table_name=t,
                    column_name=str(r[1]),
                    data_type=str(r[2]),
                    is_nullable=(str(r[3]).upper() == 'YES')
                ))
        return schema
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error introspecting schema: {str(e)}")

@router.get("/tables", response_model=List[TableInfo])
async def get_table_info():
    """Get information about all tables in the database (public schema)"""
    try:
        try:
            from ...config.database import engine
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DB engine unavailable: {e}")
        tables: List[TableInfo] = []
        with engine.connect() as conn:
            rows = conn.execute(sql_text(
                """
                SELECT c.relname AS table_name,
                       COALESCE(s.n_live_tup, 0) AS record_count,
                       pg_total_relation_size(c.oid) AS size_bytes
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                LEFT JOIN pg_stat_user_tables s ON s.relname = c.relname
                WHERE n.nspname = 'public' AND c.relkind = 'r'
                ORDER BY c.relname
                """
            )).fetchall()
            for r in rows:
                size_mb = float(r[2] or 0) / (1024.0 * 1024.0)
                tables.append(TableInfo(
                    table_name=str(r[0]),
                    record_count=int(r[1] or 0),
                    size_mb=round(size_mb, 2),
                    last_updated=datetime.now().isoformat()
                ))
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
        try:
            from ...config.database import engine
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DB engine unavailable: {e}")
        with engine.connect() as conn:
            query = sql_text(f"SELECT * FROM {table_name} LIMIT :limit OFFSET :offset")
            rows = conn.execute(query, {"limit": int(limit), "offset": int(offset)}).fetchall()
            records = [list(row) for row in rows]
        return {
            "table_name": table_name,
            "records": records,
            "total_returned": len(records),
            "limit": limit,
            "offset": offset
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting table records: {str(e)}")

@router.post("/export")
async def export_data(request: DataExportRequest, background_tasks: BackgroundTasks):
    """Export data from a specific table (json/csv using SQLAlchemy; sql via pg_dump)"""
    try:
        valid_tables = [
            'core_politician', 'bills_bill', 'hansards_statement',
            'bills_membervote', 'core_organization', 'core_membership'
        ]
        if request.table_name not in valid_tables:
            raise HTTPException(status_code=400, detail="Invalid table name")
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
        try:
            from ...config.database import engine
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DB engine unavailable: {e}")
        with engine.connect() as conn:
            row = conn.execute(sql_text(
                """
                SELECT 
                    COUNT(*) as total_politicians,
                    COUNT(DISTINCT party_name) as total_parties,
                    COUNT(DISTINCT district) as total_districts
                FROM core_politician
                """
            )).fetchone()
            analysis = {
                "total_politicians": int(row[0] or 0),
                "total_parties": int(row[1] or 0),
                "total_districts": int(row[2] or 0),
            }
        return DataAnalysisResult(
            analysis_type="politicians",
            results=analysis,
            timestamp=datetime.now().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing politicians: {str(e)}")

@router.get("/analysis/bills")
async def analyze_bills():
    """Analyze bill data"""
    try:
        try:
            from ...config.database import engine
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DB engine unavailable: {e}")
        with engine.connect() as conn:
            row = conn.execute(sql_text(
                """
                SELECT 
                    COUNT(*) as total_bills,
                    COUNT(DISTINCT session) as total_sessions,
                    COUNT(DISTINCT classification) as total_classifications
                FROM bills_bill
                """
            )).fetchone()
            analysis = {
                "total_bills": int(row[0] or 0),
                "total_sessions": int(row[1] or 0),
                "total_classifications": int(row[2] or 0),
            }
        return DataAnalysisResult(
            analysis_type="bills",
            results=analysis,
            timestamp=datetime.now().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing bills: {str(e)}")

@router.get("/analysis/hansards")
async def analyze_hansards():
    """Analyze hansard (parliamentary debate) data"""
    try:
        try:
            from ...config.database import engine
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DB engine unavailable: {e}")
        with engine.connect() as conn:
            row = conn.execute(sql_text(
                """
                SELECT 
                    COUNT(*) as total_statements,
                    COUNT(DISTINCT speaker_name) as total_speakers,
                    COUNT(DISTINCT date) as total_dates
                FROM hansards_statement
                """
            )).fetchone()
            analysis = {
                "total_statements": int(row[0] or 0),
                "total_speakers": int(row[1] or 0),
                "total_dates": int(row[2] or 0),
            }
        return DataAnalysisResult(
            analysis_type="hansards",
            results=analysis,
            timestamp=datetime.now().isoformat()
        )
    except HTTPException:
        raise
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
        valid_tables = [
            'core_politician', 'bills_bill', 'hansards_statement',
            'bills_membervote', 'core_organization', 'core_membership'
        ]
        if table_name not in valid_tables:
            raise HTTPException(status_code=400, detail="Invalid table name")
        like_param = f"%{query}%"
        if table_name == 'core_politician':
            search_query = sql_text(
                f"""
                SELECT * FROM {table_name}
                WHERE name ILIKE :q OR party_name ILIKE :q OR district ILIKE :q
                LIMIT :limit
                """
            )
        elif table_name == 'bills_bill':
            search_query = sql_text(
                f"""
                SELECT * FROM {table_name}
                WHERE title ILIKE :q OR classification ILIKE :q
                LIMIT :limit
                """
            )
        else:
            search_query = sql_text(
                f"""
                SELECT * FROM {table_name}
                WHERE CAST({"text" if table_name == 'hansards_statement' else 'id'} AS TEXT) ILIKE :q
                LIMIT :limit
                """
            )
        try:
            from ...config.database import engine
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DB engine unavailable: {e}")
        with engine.connect() as conn:
            rows = conn.execute(search_query, {"q": like_param, "limit": int(limit)}).fetchall()
            records = [list(r) for r in rows]
        return {
            "query": query,
            "table_name": table_name,
            "results": records,
            "total_found": len(records),
            "limit": limit
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching data: {str(e)}")

@router.get("/database/size")
async def get_database_size():
    """Get database size information"""
    try:
        try:
            from ...config.database import engine
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DB engine unavailable: {e}")
        with engine.connect() as conn:
            size_row = conn.execute(sql_text("SELECT pg_size_pretty(pg_database_size(current_database()));")).fetchone()
            total_size = size_row[0] if size_row and size_row[0] else "Unknown"
            sizes = {"politicians_size": "Unknown", "bills_size": "Unknown", "hansards_size": "Unknown"}
            for tbl, key in [("core_politician", "politicians_size"), ("bills_bill", "bills_size"), ("hansards_statement", "hansards_size")]:
                row = conn.execute(sql_text("SELECT pg_size_pretty(pg_total_relation_size(:t));"), {"t": tbl}).fetchone()
                if row and row[0]:
                    sizes[key] = row[0]
        return {
            "total_size": total_size,
            **sizes
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting database size: {str(e)}")

async def export_data_background(request: DataExportRequest):
    """Background task to export data using SQLAlchemy (json/csv) or pg_dump for SQL"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = os.path.join(os.getcwd(), "exports")
        os.makedirs(export_dir, exist_ok=True)
        export_file = os.path.join(export_dir, f"export_{request.table_name}_{timestamp}")
        if request.format == "csv":
            export_file += ".csv"
            try:
                from ...config.database import engine
            except Exception:
                engine = None
            if engine is None:
                return
            with engine.connect() as conn:
                rows = conn.execute(sql_text(f"SELECT * FROM {request.table_name} LIMIT :limit"), {"limit": int(request.limit or 100000)}).fetchall()
                if rows:
                    with open(export_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(rows[0].keys())
                        for row in rows:
                            writer.writerow(list(row))
        elif request.format == "json":
            export_file += ".json"
            try:
                from ...config.database import engine
            except Exception:
                engine = None
            if engine is None:
                return
            with engine.connect() as conn:
                rows = conn.execute(sql_text(f"SELECT * FROM {request.table_name} LIMIT :limit"), {"limit": int(request.limit or 100000)}).fetchall()
                data = [dict(row._mapping) for row in rows]
                with open(export_file, 'w') as f:
                    json.dump(data, f)
        else:  # sql via pg_dump (requires client inside image)
            export_file += ".sql"
            db_url = os.getenv("APP_DATABASE_URL") or os.getenv("DATABASE_URL")
            if not db_url:
                # Attempt to construct from parts
                host = os.getenv("DB_HOST", "localhost")
                port = os.getenv("DB_PORT", "5432")
                name = os.getenv("DB_NAME", "openpolicy")
                user = os.getenv("DB_USER", os.getenv("DB_USERNAME", "postgres"))
                db_url = f"postgresql://{user}@{host}:{port}/{name}"
            # Fallback to pg_dump if available
            try:
                subprocess.run([
                    "pg_dump", db_url, "-t", request.table_name, "-f", export_file
                ], capture_output=True, text=True, timeout=300)
            except Exception:
                pass
        # Log a simple marker file
        with open(os.path.join(export_dir, f"export_log_{timestamp}.log"), 'w') as f:
            f.write(f"Exported {request.table_name} as {request.format} -> {export_file}\n")
    except Exception as e:
        try:
            export_dir = os.path.join(os.getcwd(), "exports")
            os.makedirs(export_dir, exist_ok=True)
            with open(os.path.join(export_dir, f"export_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"), 'w') as f:
                f.write(f"Error exporting {request.table_name}: {str(e)}\n")
        except Exception:
            pass
