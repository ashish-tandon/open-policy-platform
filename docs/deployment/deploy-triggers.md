# Deployment Triggers

- On push to main, a repository_dispatch (`openpolicy-api-updated`) is sent to the deployment repo.
- Payload includes the built image tag (`ghcr.io/<org>/openpolicy-api:<sha>` on tagged builds) or commit SHA.

## Prereqs
- Secret `DEPLOYMENT_REPO_TOKEN` with repo:dispatch permissions for the target deployment repo
- GHCR permissions to pull images in the deploy environment

## Deployment repo
- Receives the event, pulls the image, and performs a rolling update (Kubernetes) or `docker-compose pull && up -d` (Compose)