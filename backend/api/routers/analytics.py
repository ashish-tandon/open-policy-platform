from fastapi import APIRouter

router = APIRouter()

@router.get("/status")
async def analytics_status():
    return {"service": "analytics", "status": "not-implemented", "endpoints": []}