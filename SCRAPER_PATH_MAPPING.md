# Scraper Path Mapping (Old → New)

This document keeps a definitive mapping of scraper locations from the legacy repositories to the unified architecture. Update this whenever paths are moved or renamed.

## Conventions
- Top-level `scrapers/` contains category folders:
  - `federal/` – Parliament of Canada and federal agencies
  - `provincial/` – Provincial and territorial legislatures
  - `municipal/` – Municipal governments
  - `civic/` – Civic-scraper utilities and datasets
  - `shared/` – Shared utilities and helpers
- Module/package names should use kebab-case directories and snake_case Python modules.

## Initial mappings

- `scrapers/openparliament/...` → `scrapers/federal/openparliament/...`
- `scrapers/scrapers-ca/ca_*` → `scrapers/provincial/<province>/<city_or_scope>/...`
- `scrapers/civic-scraper/...` → `scrapers/civic/...`

Add each move or rename below as it occurs:

```
<date> | <old_relative_path> → <new_relative_path> | <notes>
```

Examples:
```
2025-08-09 | scrapers/openparliament/parliament/text_analysis/views.py → scrapers/federal/openparliament/text_analysis/views.py | namespace updated
2025-08-09 | scrapers/scrapers-ca/ca_on_georgina/__init__.py → scrapers/provincial/on/georgina/__init__.py | province short code as folder
```