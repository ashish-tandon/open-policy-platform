from fastapi import APIRouter

router = APIRouter()

@router.get("/status")
async def notifications_status():
    return {"service": "notifications", "status": "not-implemented", "endpoints": []}