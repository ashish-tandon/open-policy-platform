from fastapi import APIRouter, Depends
from .health import health_check as api_health_check
from .scrapers import get_scrapers as api_get_scrapers
from ..dependencies import get_db

router = APIRouter()

@router.get("/health")
async def mobile_health():
	return await api_health_check()

@router.get("/scrapers")
async def mobile_scrapers(db=Depends(get_db)):
	# Reuse existing scrapers list
	return await api_get_scrapers(db=db)