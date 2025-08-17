# Scraper data paths and retention

## Paths
- Sources (read-only inside runner): `/scrapers`
  - Mounted from repo: `./scrapers:/scrapers:ro`
- Outputs (writable): `/app/scrapers-data`
  - Mounted from host: `./scrapers-data:/app/scrapers-data:rw`

## Structure
- Runner creates per-scraper, per-run directories:
  - `/app/scrapers-data/<scraper-name>/<YYYYMMDD_HHMMSS>/...`
- Logs for the runner live under `backend/OpenPolicyAshBack/logs/` inside the API image; compose does not currently map logs out.

## Retention
- For dev, keep last ~7 days of runs; clean older directories to save space.
- For prod, define a retention policy (e.g., 30–90 days) and a cron to prune.

## Notes
- Do not write under `/scrapers` in the container; it is read-only.
- APIs that summarize scrapers read from DB tables (`scraper_runs`, `scraper_results`) and from report files in the working directory when present.