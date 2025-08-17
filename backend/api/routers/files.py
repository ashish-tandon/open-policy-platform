from fastapi import APIRouter

router = APIRouter()

@router.get("/status")
async def files_status():
    return {"service": "files", "status": "not-implemented", "endpoints": []}