# Network & Service Discovery

- Docker (dev):
  - API: 8000
  - Web: 5173
  - Nginx: 80 → API 8000
- Kubernetes (prod):
  - In-cluster service discovery via DNS: `openpolicy-api.default.svc`
  - Ingress routes `api.yourdomain.com` → `openpolicy-api` Service → API pods
  - Probes: liveness `/api/v1/health`, readiness `/api/v1/health/detailed`
