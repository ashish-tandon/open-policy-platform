#!/usr/bin/env python3
"""
Run scrapers by category (dev):
- Record a run in scrapers DB
- Execute the dev scanner (to generate report)
- Update the run with end time, status, and records count
"""
import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from sqlalchemy import text as sql_text
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[1]


def _get_env_url() -> str:
	url = os.getenv("SCRAPERS_DATABASE_URL") or os.getenv("DATABASE_URL") or os.getenv("APP_DATABASE_URL")
	if url:
		return url
	host = os.getenv("DB_HOST", "localhost")
	port = os.getenv("DB_PORT", "5432")
	name = os.getenv("DB_NAME", "openpolicy_scrapers")
	user = os.getenv("DB_USER", os.getenv("DB_USERNAME", "postgres"))
	pwd = os.getenv("DB_PASSWORD", "")
	if pwd:
		return f"postgresql://{user}:{pwd}@{host}:{port}/{name}"
	return f"postgresql://{user}@{host}:{port}/{name}"


def _get_engine():
	return create_engine(_get_env_url(), pool_pre_ping=True)


def record_start(engine, category: str) -> int:
	with engine.begin() as conn:
		row = conn.execute(
			sql_text("INSERT INTO scraper_runs (category, status) VALUES (:c, 'started') RETURNING id"),
			{"c": category},
		).fetchone()
		return int(row[0]) if row else 0


def record_end(engine, run_id: int, status: str, records: int) -> None:
	with engine.begin() as conn:
		conn.execute(
			sql_text(
				"UPDATE scraper_runs SET end_time = NOW(), status = :st, records_collected = :rc WHERE id = :id"
			),
			{"st": status, "rc": records, "id": run_id},
		)


def latest_report_records(cwd: Path) -> int:
	files = sorted([p for p in cwd.glob("scraper_test_report_*.json")], key=lambda p: p.stat().st_mtime, reverse=True)
	if not files:
		return 0
	try:
		with open(files[0], "r") as f:
			data = json.load(f)
		return int(data.get("summary", {}).get("total_records_collected", 0))
	except Exception:
		return 0


def main() -> int:
	parser = argparse.ArgumentParser()
	parser.add_argument("--category", required=True)
	args = parser.parse_args()

	engine = _get_engine()
	category = args.category
	run_id = record_start(engine, category)

	# Execute the dev scanner for the category
	status = "completed"
	records = 0
	try:
		cmd = [
			"python",
			str(ROOT / "OpenPolicyAshBack" / "scraper_testing_framework.py"),
			"--category",
			category,
			"--max-sample-records",
			"5",
			"--verbose",
		]
		subprocess.run(cmd, cwd=str(ROOT), check=False)
		records = latest_report_records(Path.cwd())
	except Exception:
		status = "failed"
	finally:
		try:
			record_end(engine, run_id, status, records)
		except Exception:
			pass
	print(f"Run recorded: id={run_id} status={status} records={records}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())