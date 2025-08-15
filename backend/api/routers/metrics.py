from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest, Gauge
from sqlalchemy import text as sql_text

router = APIRouter()

# Custom gauges (populated on each scrape)
_latest_run_ts = Gauge("openpolicy_scraper_latest_run_timestamp_seconds", "Unix timestamp of the latest scraper run end (or start)")
_runs_total = Gauge("openpolicy_scraper_runs_total", "Total scraper runs by status", ["status"])
_latest_records = Gauge("openpolicy_scraper_latest_records", "Records collected in latest run per category", ["category"])
_latest_run_ts_by_category = Gauge("openpolicy_scraper_latest_run_category_timestamp_seconds", "Unix timestamp of latest run per category", ["category"])


def _update_scraper_metrics() -> None:
	try:
		try:
			from backend.config.database import scrapers_engine
		except Exception:
			scrapers_engine = None
		if scrapers_engine is None:
			return
		with scrapers_engine.connect() as conn:
			# Latest run timestamp
			row = conn.execute(sql_text(
				"""
				SELECT EXTRACT(EPOCH FROM COALESCE(end_time, start_time))::bigint AS ts
				FROM scraper_runs ORDER BY id DESC LIMIT 1
				"""
			)).fetchone()
			if row and row[0] is not None:
				_latest_run_ts.set(int(row[0]))
			# Runs total by status
			rows = conn.execute(sql_text(
				"""
				SELECT status, COUNT(*) FROM scraper_runs GROUP BY status
				"""
			)).fetchall()
			for r in rows:
				_runs_total.labels(status=str(r[0])).set(int(r[1] or 0))
			# Latest records per category (distinct on category)
			rows = conn.execute(sql_text(
				"""
				SELECT DISTINCT ON (category) category, COALESCE(records_collected, 0), EXTRACT(EPOCH FROM COALESCE(end_time, start_time))::bigint
				FROM scraper_runs
				WHERE end_time IS NOT NULL
				ORDER BY category, id DESC
				"""
			)).fetchall()
			for r in rows:
				_latest_records.labels(category=str(r[0])).set(int(r[1] or 0))
				if r[2] is not None:
					_latest_run_ts_by_category.labels(category=str(r[0])).set(int(r[2]))
	except Exception:
		# Best-effort; do not block metrics endpoint
		return


@router.get("/metrics")
async def metrics() -> Response:
	_update_scraper_metrics()
	return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
