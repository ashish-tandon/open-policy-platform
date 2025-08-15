# Agent Scripts

This directory contains helper scripts for local development and validation.

- `deploy_local.sh`: Builds and starts core services (postgres, api, web, scraper-runner), waits for health, initializes scrapers DB tables, then smoke tests.
- `enable_workers.sh`: Starts Redis, Celery worker, and Flower (with `workers` profile) and prints worker status.
- `check_status.sh`: Calls API health endpoints and worker status.
- `init_scrapers_db.sh`: Initializes scrapers DB tables via container or locally.
- `watch_and_test.sh`: Continuous watcher to pull updates, rebuild, test, and optionally publish results.
- `continue.sh`: Pulls latest, deploys core, shows latest results log tail.

## New: Scheduled Runs (Kubernetes)

Kubernetes CronJobs are provided in `infrastructure/k8s/cron-scrapers.yaml` to run each category daily. Apply them with:

```bash
kubectl apply -f infrastructure/k8s/cron-scrapers.yaml
```

Ensure secret `openpolicy-secrets` contains `SCRAPERS_DATABASE_URL`.

Validate:
- Inspect CronJobs: `kubectl get cronjobs`
- View last Job logs: `kubectl logs job/<job-name>`

## Validate Metrics and Entities

- Metrics:
  - `curl http://localhost:8000/metrics | grep openpolicy_scraper`
- Entities:
  - Representatives: `curl "http://localhost:8000/api/v1/entities/representatives?limit=5&q=smith"`
  - Bills: `curl "http://localhost:8000/api/v1/entities/bills?limit=5&q=tax"`
  - Committees: `curl "http://localhost:8000/api/v1/entities/committees?limit=5"`
  - Votes: `curl "http://localhost:8000/api/v1/entities/votes?limit=5&q=John"`


