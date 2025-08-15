#!/usr/bin/env bash
set -euo pipefail

OUT_FILE=${1:-analysis/current_schema.json}
DB_USER=${DB_USER:-openpolicy}
DB_PASS=${DB_PASS:-openpolicy123}
DB_DB=${DB_DB:-openpolicy_app}
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}

mkdir -p "$(dirname "$OUT_FILE")"

export PGPASSWORD="$DB_PASS"
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_DB" -At -F $'\t' <<'SQL' | python3 - <<'PY'
SELECT table_name, column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_schema = 'public'
ORDER BY table_name, ordinal_position;
SQL
import sys, json
rows=[l.strip().split('\t') for l in sys.stdin if l.strip()]
out={}
for t,c,d,n in rows:
    out.setdefault(t,[]).append({"column_name":c,"data_type":d,"is_nullable":n})
print(json.dumps(out, indent=2))
PY

