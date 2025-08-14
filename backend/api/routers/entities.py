from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text as sql_text
from typing import Any, Dict, List, Optional, Set

router = APIRouter(prefix="/api/v1/entities", tags=["Entities"])


def _engine():
	try:
		from ...config.database import engine
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"DB engine unavailable: {e}")
	return engine


def _get_columns(table: str) -> Set[str]:
	"""Return available column names for table in public schema."""
	with _engine().connect() as conn:
		rows = conn.execute(sql_text(
			"""
			SELECT column_name
			FROM information_schema.columns
			WHERE table_schema = 'public' AND table_name = :t
			"""
		), {"t": table}).fetchall()
	return {str(r[0]) for r in rows}


def _choose(cols: Set[str], candidates: List[str]) -> Optional[str]:
	for c in candidates:
		if c in cols:
			return c
	return None


def _paginate(limit: int, offset: int) -> Dict[str, int]:
	return {"limit": int(limit), "offset": int(offset)}


@router.get("/representatives")
async def list_representatives(q: Optional[str] = None, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
	"""List representatives (maps to core_politician in old schema). Supports optional search."""
	try:
		table = "core_politician"
		cols = _get_columns(table)
		id_col = _choose(cols, ["id"]) or "id"
		name_col = _choose(cols, ["name", "full_name"]) or "name"
		party_col = _choose(cols, ["party_name", "party"]) or None
		district_col = _choose(cols, ["district", "constituency"]) or None
		email_col = _choose(cols, ["email", "email_address"]) or None
		phone_col = _choose(cols, ["phone", "telephone", "phone_number"]) or None
		select_parts = [f"{id_col} AS id", f"{name_col} AS name"]
		if party_col: select_parts.append(f"{party_col} AS party")
		if district_col: select_parts.append(f"{district_col} AS district")
		if email_col: select_parts.append(f"{email_col} AS email")
		if phone_col: select_parts.append(f"{phone_col} AS phone")
		where_parts: List[str] = []
		params: Dict[str, Any] = _paginate(limit, offset)
		if q:
			params["q"] = f"%{q}%"
			search_targets = [name_col]
			if party_col: search_targets.append(party_col)
			if district_col: search_targets.append(district_col)
			where_parts.append("( " + " OR ".join([f"{c} ILIKE :q" for c in search_targets]) + " )")
		sql = f"SELECT {', '.join(select_parts)} FROM {table}"
		if where_parts:
			sql += " WHERE " + " AND ".join(where_parts)
		sql += " ORDER BY " + id_col + " ASC LIMIT :limit OFFSET :offset"
		with _engine().connect() as conn:
			rows = conn.execute(sql_text(sql), params).fetchall()
			items = [dict(r._mapping) for r in rows]
		return {"items": items, "count": len(items), "limit": limit, "offset": offset}
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Error listing representatives: {e}")


@router.get("/bills")
async def list_bills(q: Optional[str] = None, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
	"""List bills (maps to bills_bill in old schema). Supports optional search."""
	try:
		table = "bills_bill"
		cols = _get_columns(table)
		id_col = _choose(cols, ["id"]) or "id"
		title_col = _choose(cols, ["title", "name"]) or "title"
		class_col = _choose(cols, ["classification", "class"]) or None
		session_col = _choose(cols, ["session", "legislative_session"]) or None
		select_parts = [f"{id_col} AS id", f"{title_col} AS title"]
		if class_col: select_parts.append(f"{class_col} AS classification")
		if session_col: select_parts.append(f"{session_col} AS session")
		where_parts: List[str] = []
		params: Dict[str, Any] = _paginate(limit, offset)
		if q:
			params["q"] = f"%{q}%"
			targets = [title_col]
			if class_col: targets.append(class_col)
			where_parts.append("( " + " OR ".join([f"{c} ILIKE :q" for c in targets]) + " )")
		sql = f"SELECT {', '.join(select_parts)} FROM {table}"
		if where_parts:
			sql += " WHERE " + " AND ".join(where_parts)
		sql += " ORDER BY " + id_col + " DESC LIMIT :limit OFFSET :offset"
		with _engine().connect() as conn:
			rows = conn.execute(sql_text(sql), params).fetchall()
			items = [dict(r._mapping) for r in rows]
		return {"items": items, "count": len(items), "limit": limit, "offset": offset}
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Error listing bills: {e}")


@router.get("/committees")
async def list_committees(q: Optional[str] = None, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
	"""List committees (maps to core_organization in old schema). Supports optional search."""
	try:
		table = "core_organization"
		cols = _get_columns(table)
		id_col = _choose(cols, ["id"]) or "id"
		name_col = _choose(cols, ["name", "committee_name"]) or "name"
		class_col = _choose(cols, ["classification"]) or None
		select_parts = [f"{id_col} AS id", f"{name_col} AS name"]
		if class_col: select_parts.append(f"{class_col} AS classification")
		where_parts: List[str] = []
		params: Dict[str, Any] = _paginate(limit, offset)
		if q:
			params["q"] = f"%{q}%"
			targets = [name_col]
			if class_col: targets.append(class_col)
			where_parts.append("( " + " OR ".join([f"{c} ILIKE :q" for c in targets]) + " )")
		sql = f"SELECT {', '.join(select_parts)} FROM {table}"
		if where_parts:
			sql += " WHERE " + " AND ".join(where_parts)
		sql += " ORDER BY " + id_col + " ASC LIMIT :limit OFFSET :offset"
		with _engine().connect() as conn:
			rows = conn.execute(sql_text(sql), params).fetchall()
			items = [dict(r._mapping) for r in rows]
		return {"items": items, "count": len(items), "limit": limit, "offset": offset}
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Error listing committees: {e}")


@router.get("/votes")
async def list_votes(q: Optional[str] = None, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
	"""List votes (maps to bills_membervote in old schema). Supports optional search."""
	try:
		table = "bills_membervote"
		cols = _get_columns(table)
		id_col = _choose(cols, ["id"]) or "id"
		bill_col = _choose(cols, ["bill_id"]) or None
		member_col = _choose(cols, ["member_name", "person_name"]) or None
		vote_col = _choose(cols, ["vote", "option"]) or None
		select_parts = [f"{id_col} AS id"]
		if bill_col: select_parts.append(f"{bill_col} AS bill_id")
		if member_col: select_parts.append(f"{member_col} AS member")
		if vote_col: select_parts.append(f"{vote_col} AS vote")
		where_parts: List[str] = []
		params: Dict[str, Any] = _paginate(limit, offset)
		if q and member_col:
			params["q"] = f"%{q}%"
			where_parts.append(f"{member_col} ILIKE :q")
		sql = f"SELECT {', '.join(select_parts)} FROM {table}"
		if where_parts:
			sql += " WHERE " + " AND ".join(where_parts)
		sql += " ORDER BY " + id_col + " DESC LIMIT :limit OFFSET :offset"
		with _engine().connect() as conn:
			rows = conn.execute(sql_text(sql), params).fetchall()
			items = [dict(r._mapping) for r in rows]
		return {"items": items, "count": len(items), "limit": limit, "offset": offset}
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Error listing votes: {e}")