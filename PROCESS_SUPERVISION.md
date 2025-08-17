# Process Supervision and Checkpointing

This document defines the cadence, checklists, and automation for keeping implementation aligned with the architecture and documentation. It enables 15-minute reflections and 30-minute checkpoints during development without blocking delivery.

## Cadence

- 15-minute reflection: Document what changed, why, and confirm alignment with plan and architecture.
- 30-minute checkpoint: Run automated supervisor checks, verify documentation updates, and confirm tests/build are still green.

## Checklists

### 15-minute reflection checklist
- Describe edits made (files, components, endpoints).
- State alignment with architecture and Agent guidance documents.
- Note any renames or moves; update `SCRAPER_PATH_MAPPING.md` if scrapers were touched.
- Record any test changes required and why.

### 30-minute checkpoint checklist (automated + manual)
- Automated supervisor script runs:
  - Architecture conformance checks (required folders present; no hardcoded owners like personal DB usernames).
  - Backend health: smoke tests via pytest selection.
  - Frontend type-check: `tsc -b` (build optional for speed).
  - Documentation presence and staleness checks.
  - Create/update JSON and markdown reports under `reports/supervision/`.
- Manual confirmation:
  - Review generated report summary.
  - If renames occurred, ensure mapping updated.
  - If tests changed, ensure documentation reflects the change.

## How to run locally

- 15-minute reflection (manual entry):
  - Append an entry to `reports/supervision/reflections.md` using the template below.
- 30-minute checkpoint (automated):
  - `bash scripts/supervisor_check.sh --interval 30`
  - For a quick 15-minute variant: `bash scripts/supervisor_check.sh --interval 15`

## CI scheduling

A GitHub Actions workflow is provided at `.github/workflows/supervisor.yml` to run the supervisor checks on a schedule (every 30 minutes) and on pushes/PRs.

## Documentation list (kept current)
- `COMPREHENSIVE_IMPLEMENTATION_STATUS.md`
- `COMPREHENSIVE_PROJECT_STATUS_REPORT.md`
- `COMPREHENSIVE_ARCHITECTURE_PLAN.md`
- `AI_AGENT_GUIDANCE_SYSTEM.md`
- `SCRAPER_REORGANIZATION_PLAN.md`
- `SCRAPER_PATH_MAPPING.md` (this repo) – maintains old→new scraper paths

## Owner/namespace policy

- Database usernames/accounts must be organization-scoped. Defaults:
  - DB user: `openpolicy`
  - DB name: `openpolicy`
- No personal usernames (e.g., `ashishtandon`) in code, config, or scripts.
- Supervisor checks will fail if such strings are detected.

## Reflection entry template (15-minute)

```
Date/Time: <UTC timestamp>
Interval: 15m reflection
Changes: <bulleted list of edits and files>
Architecture alignment: <short confirmation + reference to sections>
Docs updated: <files updated>
Tests impacted: <what changed and status>
Risks/Next: <risks and next steps until next reflection>
```

## Checkpoint report schema (30-minute)

Supervisor writes a JSON file at `reports/supervision/checkpoint-<timestamp>.json`:

```
{
  "timestamp": "2025-08-09T12:00:00Z",
  "interval": "30m",
  "architecture": {"ok": true, "details": {...}},
  "ownership": {"ok": true, "violations": []},
  "backend_tests": {"ok": true, "summary": "N/M passed"},
  "frontend_types": {"ok": true, "summary": "tsc ok"},
  "docs": {"ok": true, "stale": []},
  "notes": "..."
}
```