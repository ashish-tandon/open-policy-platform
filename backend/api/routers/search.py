from fastapi import APIRouter

router = APIRouter()

@router.get("/status")
async def search_status():
    return {"service": "search", "status": "not-implemented", "endpoints": []}