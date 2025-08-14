Agent runbook
==============

Purpose: Provide one-command flows for the online agent to deploy, validate, and operate the stack using admin endpoints.

Primary commands
----------------

1) Deploy core services (API, Web, Postgres) and run smoke test:

```
bash scripts/agent/deploy_local.sh
```

2) Enable workers profile, then inspect worker status via admin endpoints:

```
bash scripts/agent/enable_workers.sh
bash scripts/agent/check_status.sh
```

Endpoints used
--------------
- API health: GET http://localhost:8000/api/v1/health
- Detailed health: GET http://localhost:8000/api/v1/health/detailed
- Workers ping: POST http://localhost:8000/api/v1/admin/workers/ping
- Workers status: GET http://localhost:8000/api/v1/admin/workers/status

Environment variables
---------------------
- API_URL (default http://localhost:8000)
- WEB_URL (default http://localhost:5173)
- DB_HOST (default localhost)
- DB_PORT (default 5432)


