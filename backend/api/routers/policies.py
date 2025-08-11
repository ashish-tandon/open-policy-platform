"""
Enhanced Policies Router
Provides comprehensive policy management, search, and analysis functionality
"""

from fastapi import APIRouter, Depends, Query, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import subprocess
import json
from datetime import datetime, timedelta
from pydantic import BaseModel

from ..dependencies import get_db
from ..config import settings
from ...config.database import db_config

router = APIRouter()

# Data models
class PolicyCreate(BaseModel):
    title: str
    content: str
    category: str
    jurisdiction: str
    status: str = "draft"
    tags: Optional[List[str]] = None

class PolicyUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = None
    jurisdiction: Optional[str] = None
    status: Optional[str] = None
    tags: Optional[List[str]] = None

class PolicyAnalysis(BaseModel):
    policy_id: int
    analysis_type: str
    results: Dict[str, Any]
    timestamp: str

@router.get("/")
async def get_policies(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    search: Optional[str] = None,
    category: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get policies with advanced filtering and pagination"""
    try:
        # Build query based on filters
        query = "SELECT * FROM bills_bill WHERE 1=1"
        params = []
        
        if search:
            query += " AND (title ILIKE %s OR content ILIKE %s)"
            params.extend([f"%{search}%", f"%{search}%"])
        
        if category:
            query += " AND classification = %s"
            params.append(category)
        
        if jurisdiction:
            query += " AND jurisdiction_id = %s"
            params.append(jurisdiction)
        
        if status:
            query += " AND status = %s"
            params.append(status)
        
        # Add pagination
        offset = (page - 1) * limit
        query += f" ORDER BY created_at DESC LIMIT {limit} OFFSET {offset}"
        
        # Execute query
        result = subprocess.run([
            "psql", "-h", db_config.host, "-U", db_config.username, "-d", db_config.database,
            "-c", query,
            "-t", "-A"
        ], capture_output=True, text=True, timeout=30)
        
        policies = []
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    fields = line.split('|')
                    policies.append({
                        "id": fields[0] if len(fields) > 0 else None,
                        "title": fields[1] if len(fields) > 1 else None,
                        "content": fields[2] if len(fields) > 2 else None,
                        "category": fields[3] if len(fields) > 3 else None,
                        "jurisdiction": fields[4] if len(fields) > 4 else None,
                        "status": fields[5] if len(fields) > 5 else None,
                        "created_at": fields[6] if len(fields) > 6 else None
                    })
        
        # Get total count
        count_query = "SELECT COUNT(*) FROM bills_bill"
        count_result = subprocess.run([
            "psql", "-h", db_config.host, "-U", db_config.username, "-d", db_config.database,
            "-c", count_query,
            "-t", "-A"
        ], capture_output=True, text=True, timeout=10)
        
        total = 0
        if count_result.returncode == 0 and count_result.stdout.strip():
            total = int(count_result.stdout.strip())
        
        return {
            "policies": policies,
            "total": total,
            "page": page,
            "limit": limit,
            "pages": (total + limit - 1) // limit,
            "filters": {
                "search": search,
                "category": category,
                "jurisdiction": jurisdiction,
                "status": status
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving policies: {str(e)}")

@router.get("/{policy_id}")
async def get_policy(policy_id: int, db: Session = Depends(get_db)):
    """Get specific policy by ID with detailed information"""
    try:
        result = subprocess.run([
            "psql", "-h", "localhost", "-U", "ashishtandon", "-d", "openpolicy",
            "-c", f"SELECT * FROM bills_bill WHERE id = {policy_id};",
            "-t", "-A"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            raise HTTPException(status_code=404, detail="Policy not found")
        
        lines = result.stdout.strip().split('\n')
        if not lines or not lines[0].strip():
            raise HTTPException(status_code=404, detail="Policy not found")
        
        fields = lines[0].split('|')
        policy = {
            "id": fields[0] if len(fields) > 0 else None,
            "title": fields[1] if len(fields) > 1 else None,
            "content": fields[2] if len(fields) > 2 else None,
            "category": fields[3] if len(fields) > 3 else None,
            "jurisdiction": fields[4] if len(fields) > 4 else None,
            "status": fields[5] if len(fields) > 5 else None,
            "created_at": fields[6] if len(fields) > 6 else None,
            "updated_at": fields[7] if len(fields) > 7 else None
        }
        
        return policy
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving policy: {str(e)}")

@router.get("/search/advanced")
async def search_policies_advanced(
    q: str = Query(..., min_length=1),
    category: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """Advanced policy search with multiple criteria"""
    try:
        # Build advanced search query
        query = """
        SELECT * FROM bills_bill 
        WHERE (title ILIKE %s OR content ILIKE %s OR classification ILIKE %s)
        """
        params = [f"%{q}%", f"%{q}%", f"%{q}%"]
        
        if category:
            query += " AND classification = %s"
            params.append(category)
        
        if jurisdiction:
            query += " AND jurisdiction_id = %s"
            params.append(jurisdiction)
        
        if date_from:
            query += " AND created_at >= %s"
            params.append(date_from)
        
        if date_to:
            query += " AND created_at <= %s"
            params.append(date_to)
        
        query += f" ORDER BY created_at DESC LIMIT {limit}"
        
        # Execute search
        result = subprocess.run([
            "psql", "-h", "localhost", "-U", "ashishtandon", "-d", "openpolicy",
            "-c", query,
            "-t", "-A"
        ], capture_output=True, text=True, timeout=30)
        
        policies = []
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    fields = line.split('|')
                    policies.append({
                        "id": fields[0] if len(fields) > 0 else None,
                        "title": fields[1] if len(fields) > 1 else None,
                        "content": fields[2] if len(fields) > 2 else None,
                        "category": fields[3] if len(fields) > 3 else None,
                        "jurisdiction": fields[4] if len(fields) > 4 else None,
                        "status": fields[5] if len(fields) > 5 else None,
                        "created_at": fields[6] if len(fields) > 6 else None
                    })
        
        return {
            "query": q,
            "results": policies,
            "total_found": len(policies),
            "filters": {
                "category": category,
                "jurisdiction": jurisdiction,
                "date_from": date_from,
                "date_to": date_to
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching policies: {str(e)}")

@router.get("/search")
async def search_policies(
    q: str = Query(..., min_length=1),
    db: Session = Depends(get_db)
):
    """Simple policy search"""
    return await search_policies_advanced(q=q, db=db)

@router.get("/categories")
async def get_policy_categories(db: Session = Depends(get_db)):
    """Get all available policy categories"""
    try:
        result = subprocess.run([
            "psql", "-h", "localhost", "-U", "ashishtandon", "-d", "openpolicy",
            "-c", "SELECT DISTINCT classification FROM bills_bill WHERE classification IS NOT NULL ORDER BY classification;",
            "-t", "-A"
        ], capture_output=True, text=True, timeout=10)
        
        categories = []
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    categories.append(line.strip())
        
        return {
            "categories": categories,
            "total_categories": len(categories)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving categories: {str(e)}")

@router.get("/jurisdictions")
async def get_policy_jurisdictions(db: Session = Depends(get_db)):
    """Get all available policy jurisdictions"""
    try:
        result = subprocess.run([
            "psql", "-h", "localhost", "-U", "ashishtandon", "-d", "openpolicy",
            "-c", "SELECT DISTINCT jurisdiction_id FROM bills_bill WHERE jurisdiction_id IS NOT NULL ORDER BY jurisdiction_id;",
            "-t", "-A"
        ], capture_output=True, text=True, timeout=10)
        
        jurisdictions = []
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    jurisdictions.append(line.strip())
        
        return {
            "jurisdictions": jurisdictions,
            "total_jurisdictions": len(jurisdictions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving jurisdictions: {str(e)}")

@router.get("/stats")
async def get_policy_statistics(db: Session = Depends(get_db)):
    """Get policy statistics and analytics"""
    try:
        # Total policies
        total_result = subprocess.run([
            "psql", "-h", "localhost", "-U", "ashishtandon", "-d", "openpolicy",
            "-c", "SELECT COUNT(*) FROM bills_bill;",
            "-t", "-A"
        ], capture_output=True, text=True, timeout=10)
        
        total_policies = 0
        if total_result.returncode == 0 and total_result.stdout.strip():
            total_policies = int(total_result.stdout.strip())
        
        # Categories distribution
        categories_result = subprocess.run([
            "psql", "-h", "localhost", "-U", "ashishtandon", "-d", "openpolicy",
            "-c", "SELECT classification, COUNT(*) FROM bills_bill WHERE classification IS NOT NULL GROUP BY classification ORDER BY COUNT(*) DESC;",
            "-t", "-A"
        ], capture_output=True, text=True, timeout=10)
        
        categories_dist = {}
        if categories_result.returncode == 0:
            lines = categories_result.stdout.strip().split('\n')
            for line in lines:
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 2:
                        categories_dist[parts[0].strip()] = int(parts[1].strip())
        
        # Recent policies
        recent_result = subprocess.run([
            "psql", "-h", "localhost", "-U", "ashishtandon", "-d", "openpolicy",
            "-c", "SELECT COUNT(*) FROM bills_bill WHERE created_at >= NOW() - INTERVAL '30 days';",
            "-t", "-A"
        ], capture_output=True, text=True, timeout=10)
        
        recent_policies = 0
        if recent_result.returncode == 0 and recent_result.stdout.strip():
            recent_policies = int(recent_result.stdout.strip())
        
        stats = {
            "total_policies": total_policies,
            "recent_policies_30_days": recent_policies,
            "categories_distribution": categories_dist,
            "top_categories": dict(list(categories_dist.items())[:5]),
            "timestamp": datetime.now().isoformat()
        }
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving policy statistics: {str(e)}")

@router.get("/{policy_id}/analysis")
async def analyze_policy(policy_id: int, db: Session = Depends(get_db)):
    """Analyze a specific policy"""
    try:
        # Get policy content
        policy_result = subprocess.run([
            "psql", "-h", "localhost", "-U", "ashishtandon", "-d", "openpolicy",
            "-c", f"SELECT title, content FROM bills_bill WHERE id = {policy_id};",
            "-t", "-A"
        ], capture_output=True, text=True, timeout=10)
        
        if policy_result.returncode != 0 or not policy_result.stdout.strip():
            raise HTTPException(status_code=404, detail="Policy not found")
        
        fields = policy_result.stdout.strip().split('|')
        title = fields[0] if len(fields) > 0 else ""
        content = fields[1] if len(fields) > 1 else ""
        
        # Basic text analysis
        word_count = len(content.split()) if content else 0
        char_count = len(content) if content else 0
        sentence_count = len([s for s in content.split('.') if s.strip()]) if content else 0
        
        # Category analysis
        category_result = subprocess.run([
            "psql", "-h", "localhost", "-U", "ashishtandon", "-d", "openpolicy",
            "-c", f"SELECT classification FROM bills_bill WHERE id = {policy_id};",
            "-t", "-A"
        ], capture_output=True, text=True, timeout=10)
        
        category = "Unknown"
        if category_result.returncode == 0 and category_result.stdout.strip():
            category = category_result.stdout.strip()
        
        analysis = {
            "policy_id": policy_id,
            "title": title,
            "category": category,
            "text_analysis": {
                "word_count": word_count,
                "character_count": char_count,
                "sentence_count": sentence_count,
                "average_words_per_sentence": round(word_count / sentence_count, 2) if sentence_count > 0 else 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing policy: {str(e)}")

@router.post("/")
async def create_policy(policy: PolicyCreate, db: Session = Depends(get_db)):
    """Create a new policy"""
    try:
        # Insert new policy
        insert_query = f"""
        INSERT INTO bills_bill (title, content, classification, jurisdiction_id, status, created_at)
        VALUES ('{policy.title}', '{policy.content}', '{policy.category}', '{policy.jurisdiction}', '{policy.status}', NOW())
        RETURNING id;
        """
        
        result = subprocess.run([
            "psql", "-h", "localhost", "-U", "ashishtandon", "-d", "openpolicy",
            "-c", insert_query,
            "-t", "-A"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail="Error creating policy")
        
        policy_id = result.stdout.strip()
        
        return {
            "message": "Policy created successfully",
            "policy_id": policy_id,
            "policy": policy.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating policy: {str(e)}")

@router.put("/{policy_id}")
async def update_policy(policy_id: int, policy_update: PolicyUpdate, db: Session = Depends(get_db)):
    """Update an existing policy"""
    try:
        # Build update query
        update_parts = []
        if policy_update.title:
            update_parts.append(f"title = '{policy_update.title}'")
        if policy_update.content:
            update_parts.append(f"content = '{policy_update.content}'")
        if policy_update.category:
            update_parts.append(f"classification = '{policy_update.category}'")
        if policy_update.jurisdiction:
            update_parts.append(f"jurisdiction_id = '{policy_update.jurisdiction}'")
        if policy_update.status:
            update_parts.append(f"status = '{policy_update.status}'")
        
        if not update_parts:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        update_parts.append("updated_at = NOW()")
        update_query = f"UPDATE bills_bill SET {', '.join(update_parts)} WHERE id = {policy_id};"
        
        result = subprocess.run([
            "psql", "-h", "localhost", "-U", "ashishtandon", "-d", "openpolicy",
            "-c", update_query,
            "-t", "-A"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail="Error updating policy")
        
        return {
            "message": "Policy updated successfully",
            "policy_id": policy_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating policy: {str(e)}")

@router.delete("/{policy_id}")
async def delete_policy(policy_id: int, db: Session = Depends(get_db)):
    """Delete a policy"""
    try:
        result = subprocess.run([
            "psql", "-h", "localhost", "-U", "ashishtandon", "-d", "openpolicy",
            "-c", f"DELETE FROM bills_bill WHERE id = {policy_id};",
            "-t", "-A"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail="Error deleting policy")
        
        return {
            "message": "Policy deleted successfully",
            "policy_id": policy_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting policy: {str(e)}")
