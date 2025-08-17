#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p dist
python - << 'PY'
import json, yaml
from backend.api.main import app
schema = app.openapi()
with open('dist/openapi.json', 'w') as f:
    json.dump(schema, f, indent=2)
with open('dist/openapi.yaml', 'w') as f:
    yaml.safe_dump(schema, f, sort_keys=False)
print('Wrote dist/openapi.json and dist/openapi.yaml')
PY