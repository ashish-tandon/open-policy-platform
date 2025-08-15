# Coding Guidelines

- Follow PEP8 and project style
- Use meaningful names, small functions, early returns
- Add response models to API where feasible
- Update docs in same PR as code changes

## Project structure and navigation
- Backend (FastAPI) lives under `backend/api` with entrypoint `backend.api.main:app`
- Shared config and DB utilities live under `backend/config`
- Web app lives under `web/` and reads `VITE_API_URL`
- Scripts to run/build/check live under `scripts/`
- Docs are the source of truth under `docs/`

## Imports
- In API modules, import DB primitives from `config.database`, e.g.:
  - `from config.database import engine, scrapers_engine`
- Do not use function-level imports for DB in request handlers; import at module scope
- Avoid package-relative imports that escape the top-level (no `from .. ..` for `backend.*`)

## Archiving policy (never delete)
- Do not delete files; move deprecated or replaced files to `docs/archive/` or a dedicated `archive/` folder at the appropriate level
- Add a brief README in archive folders explaining why items were archived and when

## Run instructions
- Preferred dev run: Docker Compose at repo root (`docker compose up -d api web postgres redis`)
- Local-only run:
  - Ensure Python 3.11+, install backend deps from `backend/requirements.txt`
  - Set env vars per `env.example` or `docs/operations/environment-variables.md`
  - Start API: `uvicorn backend.api.main:app --host 0.0.0.0 --port 8000`
  - Start web: `cd web && npm install && npm run dev -- --host`
- Health endpoints: `/api/v1/health`, `/api/v1/health/detailed`, `/metrics`

## Database
- Use `DATABASE_URL` as canonical; optional logical DBs: `APP_DATABASE_URL`, `SCRAPERS_DATABASE_URL`, `AUTH_DATABASE_URL`
- Root Compose creates `openpolicy_app`, `openpolicy_scrapers`, `openpolicy_auth`
- Seed data before using Entities endpoints, or expect empty lists

## Pull request hygiene
- Small, verifiable edits; keep docs updated in the same PR
- Run `bash scripts/check-docs-links.sh` and `bash scripts/export-openapi.sh` before pushing
- After changing envs or ports, update `env.example` and `docs/operations/environment-variables.md`