# Kubernetes Deployment

Apply manifests in this order:

```bash
kubectl apply -f infrastructure/k8s/api-deployment.yaml
kubectl apply -f infrastructure/k8s/api-service.yaml
kubectl apply -f infrastructure/k8s/ingress.yaml
```

Ensure secrets exist:
- `openpolicy-secrets` with keys: `DATABASE_URL`, `SECRET_KEY`

Probes:
- Liveness: `/api/v1/health`
- Readiness: `/api/v1/health/detailed`