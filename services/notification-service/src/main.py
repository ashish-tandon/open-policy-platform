from fastapi import FastAPI, Response, Query
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

app = FastAPI(title="notification-service")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/readyz")
def readyz():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/notify")
def notify(message: str = Query("")):
    return {"status": "accepted", "message": message}