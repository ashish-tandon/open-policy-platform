# Testing

- Unit/Integration: see `backend/tests`
- Smoke: `scripts/smoke-test.sh`
- OpenAPI: `scripts/export-openapi.sh`
- CI: `.github/workflows/tests.yml` and `docs-openapi.yml`

Run locally:
```bash
# Backend tests (example selection)
cd backend
pytest -q
```