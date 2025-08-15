#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p dist
python - << 'PY'
import json
import sys, os
sys.path.append(os.path.join(os.getcwd(), 'backend'))
from backend.api.main import app
schema = app.openapi()
with open('dist/openapi.json', 'w') as f:
    json.dump(schema, f, indent=2)
print('Wrote dist/openapi.json')
PY