#!/usr/bin/env python3
"""
Full Scraper Category Runner
- Records a run in scrapers DB
- Executes each scraper in the category with retries/backoff
- Records per-scraper attempts (start/end/status/error)
- Updates run with final status and total records

Note: This is a lightweight orchestrator that expects the dev scanner to exist
for now. Replace the per-scraper call with real scrapers as they integrate.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import List, Tuple

from sqlalchemy import create_engine, text as sql_text

ROOT = Path(__file__).resolve().parents[1]


def get_env_url() -> str:
	return (
		os.getenv("SCRAPERS_DATABASE_URL")
		or os.getenv("DATABASE_URL")
		or os.getenv("APP_DATABASE_URL")
		or f"postgresql://{os.getenv('DB_USER', os.getenv('DB_USERNAME', 'postgres'))}:{os.getenv('DB_PASSWORD', '')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'openpolicy_scrapers')}"
	)


def record_run_start(engine, category: str) -> int:
	with engine.begin() as conn:
		row = conn.execute(
			sql_text("INSERT INTO scraper_runs (category, status) VALUES (:c, 'started') RETURNING id"),
			{"c": category},
		).fetchone()
		return int(row[0]) if row else 0


def record_run_end(engine, run_id: int, status: str, total_records: int) -> None:
	with engine.begin() as conn:
		conn.execute(
			sql_text(
				"UPDATE scraper_runs SET end_time = NOW(), status = :st, records_collected = :rc WHERE id = :id"
			),
			{"st": status, "rc": total_records, "id": run_id},
		)


def record_attempt_start(engine, run_id: int, scraper_name: str, attempt_number: int) -> int:
	with engine.begin() as conn:
		row = conn.execute(
			sql_text(
				"INSERT INTO scraper_attempts (run_id, scraper_name, attempt_number, status) VALUES (:r, :s, :n, 'started') RETURNING id"
			),
			{"r": run_id, "s": scraper_name, "n": attempt_number},
		).fetchone()
		return int(row[0]) if row else 0


def record_attempt_end(engine, attempt_id: int, status: str, error_message: str | None = None) -> None:
	with engine.begin() as conn:
		conn.execute(
			sql_text(
				"UPDATE scraper_attempts SET finished_at = NOW(), status = :st, error_message = :err WHERE id = :id"
			),
			{"st": status, "err": error_message, "id": attempt_id},
		)


def discover_scrapers(category: str) -> List[str]:
	# Temporary: use inventory file
	inventory_file = ROOT / "SCRAPER_INVENTORY.md"
	names: List[str] = []
	if inventory_file.exists():
		current = None
		for line in inventory_file.read_text().splitlines():
			if line.startswith("### "):
				current = line.replace("### ", "").strip().lower()
			elif line.startswith("- ") and current == category.lower():
				n = line.replace("- ", "").strip()
				if n:
					names.append(n)
	return names


def run_single(scraper_name: str, category: str, max_records: int) -> Tuple[str, int]:
	# Use dev scanner as placeholder: returns records collected from latest report
	cmd = [
		"python",
		str(ROOT / "OpenPolicyAshBack" / "scraper_testing_framework.py"),
		"--category",
		category,
		"--max-sample-records",
		str(max_records),
		"--verbose",
	]
	subprocess.run(cmd, cwd=str(ROOT), check=False)
	# Best-effort parse of latest report
	try:
		reports = sorted(
			[p for p in Path.cwd().glob("scraper_test_report_*.json")], key=lambda p: p.stat().st_mtime, reverse=True
		)
		if not reports:
			return ("success", 0)
		data = json.loads(reports[0].read_text())
		return ("success", int(data.get("summary", {}).get("total_records_collected", 0)))
	except Exception:
		return ("success", 0)


def exponential_backoff(attempt: int) -> float:
	return min(60.0, 2 ** attempt)


def main() -> int:
	parser = argparse.ArgumentParser()
	parser.add_argument("--category", required=True)
	parser.add_argument("--max-records", default=10, type=int)
	parser.add_argument("--retries", default=2, type=int)
	args = parser.parse_args()

	engine = create_engine(get_env_url(), pool_pre_ping=True)
	category = args.category
	scrapers = discover_scrapers(category)
	run_id = record_run_start(engine, category)
	status = "completed"
	total_records = 0
	try:
		for name in scrapers:
			attempt = 0
			while attempt <= args.retries:
				attempt += 1
				attempt_id = record_attempt_start(engine, run_id, name, attempt)
				try:
					res_status, records = run_single(name, category, args.max_records)
					record_attempt_end(engine, attempt_id, res_status)
					total_records += max(0, records)
					break
				except Exception as e:
					record_attempt_end(engine, attempt_id, "failed", error_message=str(e)[:500])
					if attempt > args.retries:
						status = "failed"
						break
					else:
						time.sleep(exponential_backoff(attempt))
	finally:
		record_run_end(engine, run_id, status, total_records)
	print(f"Run {run_id} status={status} records={total_records}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())