# Monitoring & Analytics

## API Metrics
- Prometheus endpoint: `/metrics`
- Health: `/api/v1/health` and `/api/v1/health/detailed`

## Prometheus
- Config: `backend/monitoring/prometheus.yml`
- Alert rules: `backend/monitoring/alert.rules.yml`
- Compose service: `prometheus` in `backend/docker-compose.yml` or top-level `docker-compose.yml`

## Grafana
- Datasource: `backend/monitoring/grafana-provisioning/datasources/datasource.yml`
- Dashboards: `backend/monitoring/grafana-provisioning/dashboards/*.json`
- Compose service: `grafana` in `backend/docker-compose.yml` or top-level `docker-compose.yml`
- Access: http://localhost:3000 (default password `admin`)

## Kubernetes
- Use `infrastructure/k8s/*` for API.
- Add Prometheus Operator or custom Prometheus deployment as per cluster standards.