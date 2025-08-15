# Docker Deployment

## Ports
- API: 8000
- Web: 5173
- DB: 5432
- Redis: 6379
- Prometheus: 9090
- Grafana: 3000

## Order
1) postgres (with init SQL)
2) api (health at /api/v1/health)
3) web
4) scraper-runner (depends on postgres)

## Commands
```bash
docker compose up -d --build
bash scripts/smoke-test.sh
```