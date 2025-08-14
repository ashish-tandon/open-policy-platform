#!/usr/bin/env python3
import json
import sys
from typing import Dict, Any

# Usage:
#   python scripts/agent/schema_diff.py legacy.json current.json > analysis/schema_diff.md

def load(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def index_tables(doc: Dict[str, Any]):
    idx = {}
    for t in doc.get('tables', []):
        key = f"{t['schema']}.{t['table']}"
        cols = {c['name']: c for c in t.get('columns', [])}
        idx[key] = {
            'columns': cols,
            'pks': [pk['column'] for pk in t.get('primary_keys', [])],
            'fks': [
                (fk['column'], f"{fk['references']['schema']}.{fk['references']['table']}.{fk['references']['column']}")
                for fk in t.get('foreign_keys', [])
            ],
        }
    return idx

def main():
    if len(sys.argv) < 3:
        print('Usage: schema_diff.py <legacy_schema.json> <current_schema.json>', file=sys.stderr)
        sys.exit(1)
    legacy = load(sys.argv[1])
    current = load(sys.argv[2])
    lidx = index_tables(legacy)
    cidx = index_tables(current)

    legacy_tables = set(lidx.keys())
    current_tables = set(cidx.keys())

    only_in_legacy = sorted(legacy_tables - current_tables)
    only_in_current = sorted(current_tables - legacy_tables)
    common = sorted(legacy_tables & current_tables)

    out = []
    out.append('# Schema Diff\n')
    if only_in_legacy:
        out.append('## Tables only in legacy\n')
        for t in only_in_legacy:
            out.append(f'- {t}')
        out.append('')
    if only_in_current:
        out.append('## Tables only in current\n')
        for t in only_in_current:
            out.append(f'- {t}')
        out.append('')

    for t in common:
        l = lidx[t]
        c = cidx[t]
        lcols = set(l['columns'].keys())
        ccols = set(c['columns'].keys())
        added = sorted(ccols - lcols)
        removed = sorted(lcols - ccols)
        changed = []
        for col in sorted(lcols & ccols):
            lc = l['columns'][col]
            cc = c['columns'][col]
            diffs = []
            if lc.get('data_type') != cc.get('data_type'):
                diffs.append(f"type {lc.get('data_type')} -> {cc.get('data_type')}")
            if bool(lc.get('is_nullable')) != bool(cc.get('is_nullable')):
                diffs.append(f"nullable {lc.get('is_nullable')} -> {cc.get('is_nullable')}")
            if (lc.get('default') or '') != (cc.get('default') or ''):
                diffs.append('default changed')
            if diffs:
                changed.append((col, ', '.join(diffs)))
        if added or removed or changed:
            out.append(f'## {t}')
            if added:
                out.append('### Added columns')
                for ccol in added:
                    out.append(f'- {ccol} ({c["columns"][ccol].get("data_type")})')
            if removed:
                out.append('### Removed columns')
                for lcol in removed:
                    out.append(f'- {lcol} ({l["columns"][lcol].get("data_type")})')
            if changed:
                out.append('### Changed columns')
                for col, d in changed:
                    out.append(f'- {col}: {d}')
            out.append('')

    sys.stdout.write('\n'.join(out))

if __name__ == '__main__':
    main()


