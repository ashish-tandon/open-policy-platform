# TODO (High-level Plan)

- Scraper framework (full)
  - Replace dev scanner with categorized execution (timeouts/retries/backoff)
  - Write outputs to DB (openpolicy_scrapers) via stable tables
  - Scheduling: daily/weekly/monthly/continuous per `backend/config/scraper_plan.py`
  - Tests: per-scraper smoke and output contract validation

- Data model and storage
  - Canonical tables for bills/representatives/committees/events/votes
  - Migrations and indexes; backup/restore scripts

- API
  - Public endpoints with filters/sort/pagination
  - Admin controls for queue/jobs/logs; data quality endpoints
  - Mobile shim expansion as needed

- UI
  - Normal UI pages: Federal/Provincial/City lists
  - Admin controls: run queue view, logs surfacing, data quality dashboards

- Observability & ops
  - Metrics and dashboards (API latency, DB health, scraper success)
  - Alerting rules for failures/health degradation

- CI/CD & deployment
  - Enforce schemas/ports in CI; scrape report validation
  - Image push + repository_dispatch + k8s deploy (wired)
  - Extend k8s to workers/gateway (optional)

- Security & performance
  - Enforce prod guards (SECRET_KEY, ALLOWED_HOSTS/ORIGINS)
  - Load testing and DB index review; caching hot endpoints