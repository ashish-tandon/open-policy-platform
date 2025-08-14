from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text as sql_text
from typing import Any, Dict

router = APIRouter(prefix="/api/v1/entities", tags=["Entities"])


def _engine():
	try:
		from ...config.database import engine
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"DB engine unavailable: {e}")
	return engine


@router.get("/representatives")
async def list_representatives(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
	"""List representatives (maps to core_politician in old schema)"""
	try:
		with _engine().connect() as conn:
			rows = conn.execute(sql_text(
				"""
				SELECT id, name, party_name, district, email, phone FROM core_politician
				ORDER BY id ASC
				LIMIT :limit OFFSET :offset
				"""
			), {"limit": int(limit), "offset": int(offset)}).fetchall()
			items = [
				{
					"id": int(r[0]),
					"name": r[1],
					"party": r[2],
					"district": r[3],
					"email": r[4],
					"phone": r[5],
				}
				for r in rows
			]
		return {"items": items, "count": len(items), "limit": limit, "offset": offset}
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Error listing representatives: {e}")


@router.get("/bills")
async def list_bills(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
	"""List bills (maps to bills_bill in old schema)"""
	try:
		with _engine().connect() as conn:
			rows = conn.execute(sql_text(
				"""
				SELECT id, title, classification, session FROM bills_bill
				ORDER BY id DESC
				LIMIT :limit OFFSET :offset
				"""
			), {"limit": int(limit), "offset": int(offset)}).fetchall()
			items = [
				{
					"id": int(r[0]),
					"title": r[1],
					"classification": r[2],
					"session": r[3],
				}
				for r in rows
			]
		return {"items": items, "count": len(items), "limit": limit, "offset": offset}
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Error listing bills: {e}")