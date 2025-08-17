# Environment Variables Mapping

Source of truth: `env.example` and `backend/api/config.py`

## Backend API (FastAPI)
- REQUIRED
  - `DATABASE_URL` (Settings.database_url) — e.g., `postgresql://user@host:5432/db`
  - `SECRET_KEY` (Settings.secret_key) — JWT signing secret (must be strong in prod)
  - `API_HOST` (Settings.host) — default `0.0.0.0`
  - `API_PORT` (Settings.port) — default `8000`
  - `ALLOWED_ORIGINS` (Settings.allowed_origins) — comma-separated; lock down in prod
  - `ALLOWED_HOSTS` (Settings.allowed_hosts) — comma-separated; lock down in prod
- OPTIONAL
  - `APP_DATABASE_URL` — preferred DB for app services if set (falls back to `DATABASE_URL`)
  - `SCRAPERS_DATABASE_URL` — preferred DB for scrapers if set (falls back to `DATABASE_URL`)
  - `AUTH_DATABASE_URL` — preferred DB for auth if set (falls back to `DATABASE_URL`)
  - `ALGORITHM` (Settings.algorithm) — default `HS256`
  - `ACCESS_TOKEN_EXPIRE_MINUTES` (Settings.access_token_expire_minutes) — default `30`
  - `REDIS_URL` (Settings.redis_url) — default `redis://localhost:6379`
  - `OPENAI_API_KEY` (Settings.openai_api_key) — only if AI features used
  - `SCRAPER_TIMEOUT` (Settings.scraper_timeout) — default `30`
  - `MAX_CONCURRENT_SCRAPERS` (Settings.max_concurrent_scrapers) — default `5`
  - `LOG_LEVEL` (Settings.log_level) — default `INFO`
  - `MAX_FILE_SIZE` (Settings.max_file_size) — default `10485760`
  - `UPLOAD_DIR` (Settings.upload_dir) — default `uploads`

Notes:
- In production: `DEBUG` off (omit or set to false), restrict CORS/hosts, set strong `SECRET_KEY`.

## Frontend (React + Vite)
- REQUIRED
  - `VITE_API_URL` — base API URL, e.g., `http://localhost:8000`
- OPTIONAL
  - `VITE_ENVIRONMENT` — e.g., `development`/`production`
  - `VITE_ENABLE_ANALYTICS` — `true/false`
  - `VITE_ENABLE_DEBUG` — `true/false`

## Database (PostgreSQL)
- Use `DATABASE_URL` for app connectivity
- Alternative granular vars for tooling/scripts (optional): `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USERNAME`, `DB_PASSWORD`
- Optional logical DB URLs for separation of concerns: `APP_DATABASE_URL`, `SCRAPERS_DATABASE_URL`, `AUTH_DATABASE_URL`
- Tests: `TEST_DATABASE_URL` (defaults to `postgresql://postgres@localhost:5432/openpolicy_test`)

## Scrapers (integration)
- Reports/logs read by API from working directory
- OPTIONAL (recommended for production organization; current code defaults to repo root)
  - `SCRAPER_REPORTS_DIR` — where `scraper_test_report_*.json` reside
  - `SCRAPER_LOGS_DIR` — where `*.log` reside

## CI/CD
- None required by default; workflow installs Python and requirements then runs scripts

## Environments
- Development: minimal set (DATABASE_URL to local, VITE_API_URL to local, permissive CORS)
- Staging/Production: set all REQUIRED; restrict CORS/hosts; secrets and managed DB

Validation
- On deploy, verify all REQUIRED variables are set for target service
- Startup guard enforces required variables; in production, missing or insecure values will fail startup