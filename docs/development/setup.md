# Developer setup

## Prereqs
- Docker Desktop + Compose v2
- Node 18+ for web; Python 3.11+ if running locally without Docker
- Git access to `ashish-tandon/open-policy-platform`

## Clone and branch workflow
- Work off `main`. Create feature branches like `feature/<short-desc>`.
- Open PRs early; keep them small; rebase on `main` before merge; use squash merge.
- Coordinate in the PR conversation; keep a live checklist there.

## Environment
- Optional `.env` for dev (compose has defaults):
  - `SECRET_KEY`, `DATABASE_URL`, `APP_DATABASE_URL`, `SCRAPERS_DATABASE_URL`, `AUTH_DATABASE_URL`
  - `REDIS_URL=redis://redis:6379/0` (default in compose)

## Run with Docker (preferred)
- Bring up core stack:
```bash
docker compose up -d --build postgres api web redis
```
- Seed minimal data (for Entities and health to show non-empty):
```bash
bash scripts/seed_db.sh
```
- Health checks:
```bash
curl -s http://localhost:8000/api/v1/health
curl -s http://localhost:8000/api/v1/health/detailed
curl -s http://localhost:8000/metrics | head
curl -s 'http://localhost:8000/api/v1/entities/representatives?limit=5'
```
- Workers (Celery + Flower):
```bash
docker compose --profile workers up -d
docker compose logs celery-worker --tail=80
docker compose logs celery-beat --tail=80
# Flower UI
open http://localhost:5555 || xdg-open http://localhost:5555 || true
```

## Scraper-runner
- Sources are mounted read-only: `./scrapers:/scrapers:ro`.
- Writable data volume for outputs: `./scrapers-data:/app/scrapers-data:rw` (already configured).
- Scrapers must write under `/app/scrapers-data/...`; the runner stages sources and syncs artifacts, without using `--output-dir`. 
- Start runner:
```bash
docker compose up -d scraper-runner
docker compose logs scraper-runner --tail=120
```

## Local (no Docker)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
# Web
(cd web && npm install && npm run dev -- --host)
```

## Code conventions
- API routers import DB engines at module scope:
  - `from config.database import engine, scrapers_engine`
- Do not import engines inside request handlers.
- Health DB probe example:
```python
from sqlalchemy import text as sql_text
from config.database import engine
with engine.connect() as conn:
    conn.execute(sql_text("SELECT 1"))
```

## OpenAPI export
```bash
bash scripts/export-openapi.sh
```
- If running locally, ensure venv is active; the script appends `backend` to `PYTHONPATH`.

## Smoke test
```bash
bash scripts/smoke.sh
```

## Communication
- Use the PR thread to post status updates, checklists, and next steps.
- Note when to merge, what to seed/import, and any failing checks.

## Common pitfalls
- Empty Entities: run `bash scripts/seed_db.sh`.
- Health “unhealthy”: ensure absolute imports (`config.database`) and no handler-scoped imports.
- Scraper-runner write errors: ensure `./scrapers-data:/scrapers/data:rw` and write under `/scrapers/data`.
- Flower logs may mention AMQP; Celery is configured to Redis for the worker/beat services.

## Near-term tasks
- Replace sample seed with real import scripts/datasets as they become available.
- Wire `scripts/smoke.sh` to CI on PRs.
- Document scraper data retention in `docs/operations/scraper-data.md`.