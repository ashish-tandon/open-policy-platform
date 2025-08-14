# Agent Scripts

One-command local workflows for the core stack and optional profiles.

## Deploy core stack and run smoke test
```bash
bash scripts/agent/deploy_local.sh
```

## Enable workers profile and validate
```bash
bash scripts/agent/enable_workers.sh
bash scripts/agent/check_status.sh
```

## Notes
- Docker Desktop must be running
- `.env` is auto-created; a `SECRET_KEY` is generated if missing
- Profiles (opt-in): `workers` (celery/flower), `ops` (prom/grafana), `gateway` (nginx)


