# Monitoring & Analytics

## API Metrics
- Prometheus endpoint: `/metrics`
- Health: `/api/v1/health` and `/api/v1/health/detailed`

## Prometheus
- Config file: `backend/monitoring/prometheus.yml`
- Run via docker-compose (service: `prometheus`)

## Grafana
- Provisioned datasource: `backend/monitoring/grafana-provisioning/datasources/datasource.yml`
- Access: http://localhost:3000 (default password `admin`)