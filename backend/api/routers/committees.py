from fastapi import APIRouter

router = APIRouter()

@router.get("/status")
async def committees_status():
    return {"service": "committees", "status": "not-implemented", "endpoints": []}