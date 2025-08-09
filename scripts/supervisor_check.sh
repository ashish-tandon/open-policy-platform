#!/usr/bin/env bash
set -euo pipefail

INTERVAL=${1:-}
if [[ "$INTERVAL" == "--interval" ]]; then
  INTERVAL="$2"
else
  INTERVAL="manual"
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPORT_DIR="$ROOT_DIR/reports/supervision"
mkdir -p "$REPORT_DIR"
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
JSON_REPORT="$REPORT_DIR/checkpoint-$TS.json"
MD_REPORT="$REPORT_DIR/checkpoint-$TS.md"

ok_arch=true
ok_owner=true
ok_backend=true
ok_frontend=true

notes=()

# 1) Architecture presence checks
required=(
  "$ROOT_DIR/backend/api"
  "$ROOT_DIR/web/src"
  "$ROOT_DIR/scrapers"
  "$ROOT_DIR/PROCESS_SUPERVISION.md"
  "$ROOT_DIR/SCRAPER_PATH_MAPPING.md"
)
for p in "${required[@]}"; do
  if [[ ! -e "$p" ]]; then
    ok_arch=false
    notes+=("missing:$p")
  fi
done

# 2) Ownership policy checks (no personal usernames)
violations=()
if grep -RInE "ashishtandon|ubuntu@|personal" "$ROOT_DIR" --exclude-dir .git --exclude-dir node_modules --exclude-dir __pycache__ >/tmp/viol 2>/dev/null; then
  ok_owner=false
  mapfile -t vlines </tmp/viol
  violations+=("${vlines[@]}")
fi

# 3) Backend smoke tests (if pytest ini exists)
if [[ -f "$ROOT_DIR/backend/pytest.ini" ]]; then
  pushd "$ROOT_DIR/backend" >/dev/null
  if ! python3 -c "import fastapi" 2>/dev/null; then
    ok_backend=false
    notes+=("python_deps_missing:fastapi")
  else
    if ! pytest -q tests/infrastructure -k health --maxfail=1 --disable-warnings; then
      ok_backend=false
      notes+=("backend_smoke_failed")
    fi
  fi
  popd >/dev/null
else
  notes+=("pytest_ini_missing")
fi

# 4) Frontend type-check
pushd "$ROOT_DIR/web" >/dev/null
if ! npm -v >/dev/null 2>&1; then
  ok_frontend=false
  notes+=("npm_missing")
else
  if ! npx -y tsc -b >/dev/null 2>&1; then
    ok_frontend=false
    notes+=("tsc_failed")
  fi
fi
popd >/dev/null

# Write JSON report
{
  echo "{"
  echo "  \"timestamp\": \"$TS\","
  echo "  \"interval\": \"$INTERVAL\","
  echo "  \"architecture\": {\"ok\": $ok_arch},"
  echo "  \"ownership\": {\"ok\": $ok_owner, \"violations\": ["
  if [[ ${#violations[@]} -gt 0 ]]; then
    for i in "${!violations[@]}"; do
      printf '    %s"%s"%s\n' '"' "${violations[$i]//"/\"}" '"' \
        | sed 's/$/,/'
    done | sed '$ s/,$//'
  fi
  echo "  ]},"
  echo "  \"backend_tests\": {\"ok\": $ok_backend},"
  echo "  \"frontend_types\": {\"ok\": $ok_frontend},"
  echo "  \"notes\": ["
  for i in "${!notes[@]}"; do
    printf '    %s"%s"%s\n' '"' "${notes[$i]}" '"' \
      | sed 's/$/,/'
  done | sed '$ s/,$//'
  echo "  ]"
  echo "}"
} >"$JSON_REPORT"

# Write Markdown summary
{
  echo "# Supervisor Checkpoint"
  echo "- **timestamp**: $TS"
  echo "- **interval**: $INTERVAL"
  echo "- **architecture ok**: $ok_arch"
  echo "- **ownership ok**: $ok_owner"
  echo "- **backend tests ok**: $ok_backend"
  echo "- **frontend types ok**: $ok_frontend"
  if [[ ${#violations[@]} -gt 0 ]]; then
    echo "\n## Ownership Violations"
    printf -- "- %s\n" "${violations[@]}"
  fi
  if [[ ${#notes[@]} -gt 0 ]]; then
    echo "\n## Notes"
    printf -- "- %s\n" "${notes[@]}"
  fi
} >"$MD_REPORT"

echo "Wrote reports to $REPORT_DIR"

# Exit non-zero if any critical check failed
if [[ "$ok_arch" != true || "$ok_owner" != true || "$ok_backend" != true || "$ok_frontend" != true ]]; then
  exit 1
fi