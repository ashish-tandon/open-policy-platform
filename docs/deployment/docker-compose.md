# Docker Compose Deployment

- API: 8000
- Web: 5173
- Postgres: 5432
- Redis: 6379
- Prometheus: 9090
- Grafana: 3000

A Postgres init file creates logical DBs: `openpolicy_app`, `openpolicy_scrapers`, `openpolicy_auth`.

Scraper Runner is a dedicated container that schedules category runs using `background_scraper_execution.py`.

Start:

```bash
docker compose up -d --build
bash scripts/smoke-test.sh
```