#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/agent/dump_schema.sh <database_name> <output_json>
# Example:
#   bash scripts/agent/dump_schema.sh openparliament_public analysis/legacy_schema.json

if [ $# -lt 2 ]; then
  echo "Usage: $0 <database_name> <output_json>" >&2
  exit 1
fi

DB_NAME="$1"
OUT_JSON="$2"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# Ensure Postgres container name
PSQL_CONT=$(docker ps --format '{{.Names}}' | grep -m1 '^open-policy-platform-postgres-1$' || true)
PSQL_USER=${PSQL_USER:-openpolicy}
if [ -z "$PSQL_CONT" ]; then
  echo "Postgres container not found" >&2
  exit 1
fi

cols_sql='SELECT table_schema, table_name, column_name, data_type, is_nullable, COALESCE(column_default, '''') as column_default, ordinal_position FROM information_schema.columns WHERE table_schema NOT IN (''pg_catalog'',''information_schema'') ORDER BY table_schema, table_name, ordinal_position;'
pk_sql="SELECT tc.table_schema, tc.table_name, kc.column_name, tc.constraint_name FROM information_schema.table_constraints tc JOIN information_schema.key_column_usage kc ON tc.constraint_name = kc.constraint_name AND tc.table_schema = kc.table_schema AND tc.table_name = kc.table_name WHERE tc.constraint_type = 'PRIMARY KEY' AND tc.table_schema NOT IN ('pg_catalog','information_schema') ORDER BY tc.table_schema, tc.table_name, kc.ordinal_position;"
fk_sql="SELECT tc.table_schema, tc.table_name, kcu.column_name, ccu.table_schema AS foreign_table_schema, ccu.table_name AS foreign_table_name, ccu.column_name AS foreign_column_name, tc.constraint_name FROM information_schema.table_constraints tc JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema JOIN information_schema.constraint_column_usage ccu ON ccu.constraint_name = tc.constraint_name WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema NOT IN ('pg_catalog','information_schema') ORDER BY tc.table_schema, tc.table_name, kcu.ordinal_position;"

docker exec -i "$PSQL_CONT" psql -U "$PSQL_USER" -d "$DB_NAME" -A -F $'\t' -t -c "$cols_sql" > "$TMP_DIR/columns.tsv" || true
docker exec -i "$PSQL_CONT" psql -U "$PSQL_USER" -d "$DB_NAME" -A -F $'\t' -t -c "$pk_sql" > "$TMP_DIR/pk.tsv" || true
docker exec -i "$PSQL_CONT" psql -U "$PSQL_USER" -d "$DB_NAME" -A -F $'\t' -t -c "$fk_sql" > "$TMP_DIR/fk.tsv" || true

python3 - "$TMP_DIR" "$OUT_JSON" << 'PY'
import json, os, sys
tmp_dir, out_json = sys.argv[1], sys.argv[2]

def read_tsv(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            rows.append(line.split('\t'))
    return rows

columns = read_tsv(os.path.join(tmp_dir, 'columns.tsv'))
pks = read_tsv(os.path.join(tmp_dir, 'pk.tsv'))
fks = read_tsv(os.path.join(tmp_dir, 'fk.tsv'))

tables = {}
for schema, table, col, dtype, nullable, default, pos in columns:
    key = f"{schema}.{table}"
    tables.setdefault(key, {"schema": schema, "table": table, "columns": [], "primary_keys": [], "foreign_keys": []})
    tables[key]["columns"].append({
        "name": col,
        "data_type": dtype,
        "is_nullable": nullable == 'YES',
        "default": default,
        "ordinal_position": int(pos)
    })

for schema, table, col, cname in pks:
    key = f"{schema}.{table}"
    if key in tables:
        tables[key]["primary_keys"].append({"column": col, "constraint_name": cname})

for schema, table, col, fschema, ftable, fcol, cname in fks:
    key = f"{schema}.{table}"
    if key in tables:
        tables[key]["foreign_keys"].append({
            "column": col,
            "references": {"schema": fschema, "table": ftable, "column": fcol},
            "constraint_name": cname
        })

result = {
    "tables": sorted(tables.values(), key=lambda t: (t["schema"], t["table"]))
}

os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print(f"Wrote schema to {out_json}")
PY

echo "Schema exported to $OUT_JSON"

