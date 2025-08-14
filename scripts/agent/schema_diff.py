#!/usr/bin/env python3
import json
import sys
from typing import Dict, List

if len(sys.argv) < 3:
	print("Usage: schema_diff.py legacy.json current.json", file=sys.stderr)
	sys.exit(1)

legacy_path, current_path = sys.argv[1], sys.argv[2]
legacy: Dict[str, List[Dict]] = json.load(open(legacy_path))
current: Dict[str, List[Dict]] = json.load(open(current_path))

out_lines = []
all_tables = sorted(set(legacy.keys()) | set(current.keys()))
for t in all_tables:
	lcols = {c['column_name']: c for c in legacy.get(t, [])}
	ccols = {c['column_name']: c for c in current.get(t, [])}
	added = sorted(set(ccols) - set(lcols))
	removed = sorted(set(lcols) - set(ccols))
	changed = []
	for col in sorted(set(lcols) & set(ccols)):
		l, c = lcols[col], ccols[col]
		if (l.get('data_type') != c.get('data_type')) or (l.get('is_nullable') != c.get('is_nullable')):
			changed.append((col, l.get('data_type'), c.get('data_type'), l.get('is_nullable'), c.get('is_nullable')))
	if added or removed or changed:
		out_lines.append(f"## Table: {t}")
		if added:
			out_lines.append("- Added columns: " + ", ".join(added))
		if removed:
			out_lines.append("- Removed columns: " + ", ".join(removed))
		for col, ld, cd, ln, cn in changed:
			out_lines.append(f"- Changed {col}: type {ld} -> {cd}, nullable {ln} -> {cn}")
		out_lines.append("")

print("\n".join(out_lines) or "No schema differences detected.")


