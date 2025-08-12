from fastapi import APIRouter

router = APIRouter()

@router.get("/status")
async def votes_status():
    return {"service": "votes", "status": "not-implemented", "endpoints": []}