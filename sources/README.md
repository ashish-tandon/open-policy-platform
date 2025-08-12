# Open Policy Platform - Unified Deployment

This workspace organizes multiple upstream repositories into a single, integrated deployment. All scrapers and services (except the mobile app) are integrated, scheduled, and documented here.

## Contents

- `sources/openparliament`: Django site and scrapers for Parliament of Canada (AGPLv3)
- `sources/scrapers-ca`: OCD Pupa scrapers for Canada (MIT)
- `sources/civic-scraper`: Python civic scraping utilities (MIT)
- `sources/OpenPolicyAshBack/OpenPolicyMerge`: Unified FastAPI API, database, scheduler, and single-container orchestration
- `sources/admin-open-policy`: Admin SPA (Vite + React)
- `sources/open-policy-infra`: Infrastructure as code (reference)
- `sources/open-policy-app`: Mobile app (deferred)

## Architecture (Organized)

- API: FastAPI (`OpenPolicyMerge/src/api/main.py`)
- Database: PostgreSQL 16, SQLAlchemy models (`OpenPolicyMerge/src/database`)
- Cache/Queue: Redis 7
- Workers: Celery (workers + beat), defined in `OpenPolicyMerge/src/workers/celery_app.py`
- Orchestration: Docker Compose (`OpenPolicyMerge/docker-compose.yml`), Supervisor inside the app container (`docker/start.sh`)
- Scrapers: Unified manager adapters for Parliament, Represent API, and municipal/provincial sites (`OpenPolicyMerge/src/scrapers`)

## Prerequisites

- Docker 24+, Docker Compose v2
- 6 free ports: 80, 8000, 5432, 6379, 5555, 3000

## One-command Run

```
cd sources/OpenPolicyAshBack/OpenPolicyMerge
docker compose up -d --build
```

- Web (static + API proxy): `http://localhost/`
- API Docs: `http://localhost:8000/docs`
- Celery Flower: `http://localhost:5555`
- Postgres: `localhost:5432` (user: `openpolicy`, pass: `secure_password`)
- Redis: `localhost:6379`

## Scheduling

Celery Beat schedules:
- Daily comprehensive scrape at 02:00 local time
- Hourly health check

Manual triggers:
```
docker compose exec openpolicy-merge-app celery -A src.workers.celery_app call src.workers.celery_app.run_comprehensive_scrape | cat
```

## Environment (defaults)

Set via `docker-compose.yml` in `OpenPolicyMerge`:
- `DATABASE_URL=postgresql://openpolicy:secure_password@db:5432/openpolicy_merge`
- `REDIS_URL=redis://redis:6379/0`
- `SECRET_KEY` (change in production)
- `CORS_ORIGINS` and `ALLOWED_HOSTS`

## Operations Runbook

- Check health: `GET /health`, `GET /stats`, Flower UI
- Logs: `docker compose logs -f openpolicy-merge` (app), `db`, `redis`
- Restart: `docker compose restart`
- Scale workers: edit `docker/start.sh` concurrency or add more worker programs
- Backups: snapshot `postgres_data` volume; export data via API as needed

## Security Notes

- Replace all example passwords (`secure_password`) and `SECRET_KEY`
- Restrict network exposure in Compose or deploy behind a reverse proxy with TLS

## Ingestion Validation

- Parliament scraping endpoints exist in the unified manager; Represent API adapter included
- Municipal/provincial adapters follow scrapers-ca patterns; extend configs per jurisdiction
- Celery Beat ensures runs occur on schedule; monitor via Flower

## Licenses and Attribution

- `openparliament` is AGPLv3: deployments must provide source for network users
- Respect terms for data sources and robots.txt; use reasonable rate limits

## Next Steps

- Wire `admin-open-policy` build into Nginx static dir if needed
- Mobile app (`open-policy-app`) deferred until later