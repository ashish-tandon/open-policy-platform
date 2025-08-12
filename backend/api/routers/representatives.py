from fastapi import APIRouter

router = APIRouter()

@router.get("/status")
async def representatives_status():
    return {"service": "representatives", "status": "not-implemented", "endpoints": []}