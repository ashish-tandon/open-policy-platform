# Cursor Rules and Developer Workflow

This project uses `.cursorrules` to guide agents and developers.

## Workflow Summary
- Read `docs/architecture/master-execution-plan.md` first
- Implement changes incrementally; keep docs as single source of truth
- Validate with:
  - `bash scripts/check-docs-links.sh`
  - `bash scripts/export-openapi.sh`
- Start local services with `./scripts/start-all.sh`

## Key Conventions
- Backend entrypoint: `backend.api.main:app` on port 8000
- Health endpoints: `/api/v1/health` (liveness) and `/api/v1/health/detailed` (readiness)
- Databases:
  - `DATABASE_URL` as canonical fallback
  - Optional: `APP_DATABASE_URL`, `SCRAPERS_DATABASE_URL`, `AUTH_DATABASE_URL`

## After Any Change
- Update relevant docs under `docs/`
- Re-run docs and OpenAPI checks
- Ensure env changes are reflected in `env.example`