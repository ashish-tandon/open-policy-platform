from fastapi import APIRouter

router = APIRouter()

@router.get("/status")
async def debates_status():
    return {"service": "debates", "status": "not-implemented", "endpoints": []}