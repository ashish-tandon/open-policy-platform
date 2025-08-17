from fastapi import FastAPI, Response
from fastapi import HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

app = FastAPI(title="config-service")
_store = {}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/readyz")
def readyz():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/config/{key}")
def get_config(key: str):
    if key not in _store:
        raise HTTPException(status_code=404, detail="not found")
    return {"key": key, "value": _store[key]}

@app.post("/config/{key}")
def set_config(key: str, value: str):
    _store[key] = value
    return {"status": "ok", "key": key, "value": value}