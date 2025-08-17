# Development Setup

1) Clone and install
```bash
git clone <repo>
cd open-policy-platform
./scripts/setup-unified.sh
```

2) Run locally
```bash
./scripts/start-all.sh
# API: http://localhost:8000
# Web: http://localhost:5173
```

3) Validate
```bash
bash scripts/check-docs-links.sh
bash scripts/export-openapi.sh
bash scripts/smoke-test.sh
```