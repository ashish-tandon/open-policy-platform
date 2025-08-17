# Deployment Process

## Development
- Prereqs: Python 3.11, Node 18+, Postgres 14+
- Setup: `./scripts/setup-unified.sh`
- Start: `./scripts/start-all.sh`

## Production
1. Provision Postgres; ensure databases exist (compose includes init SQL)
2. Set `.env` with `DATABASE_URL` or logical DB URLs and security settings
3. Build and run containers: `docker compose -f backend/docker-compose.yml up -d --build`
4. Verify health:
   - `/api/v1/health` (liveness)
   - `/api/v1/health/detailed` (readiness)
5. Export OpenAPI: `bash scripts/export-openapi.sh`

## Startup Order
- DB → API → nginx → Web

## Notes
- No secrets in code. All configuration via env.