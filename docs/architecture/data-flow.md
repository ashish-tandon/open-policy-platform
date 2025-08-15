# Data Flow (Scraper → DB → API → UI)

- Scrapers (runner): `backend/OpenPolicyAshBack/background_scraper_execution.py`
  - Categories: parliamentary, provincial, municipal, civic, update
  - Writes reports: `scraper_test_report_*.json`
  - Target DB: `openpolicy_scrapers`

- Database
  - PostgreSQL (multi-DB via init SQL)
  - Access via SQLAlchemy engine in `backend/config/database.py`

- API (FastAPI)
  - Endpoints:
    - `/api/v1/scrapers` (list, status)
    - `/api/v1/scrapers/categories` (category summary)
    - `/api/v1/health/scrapers` (aggregated health)
  - Reads latest report JSON and DB metrics

- UI (Web)
  - Pages: `web/src/pages/admin/scrapers.tsx`, `web/src/pages/public/admin/scrapers.tsx`
  - Calls API endpoints and renders categories + scrapers tables