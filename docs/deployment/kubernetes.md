# Kubernetes Deployment

Apply manifests:
```bash
kubectl apply -f infrastructure/k8s/api-deployment.yaml
kubectl apply -f infrastructure/k8s/api-service.yaml
kubectl apply -f infrastructure/k8s/ingress.yaml
```

- Readiness: `/api/v1/health` on port 8000
- Set secrets: `openpolicy-secrets` with `DATABASE_URL` and `SECRET_KEY`