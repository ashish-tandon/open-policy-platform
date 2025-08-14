import os
from celery import Celery
from datetime import timedelta
import subprocess

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
	"openpolicy",
	broker=REDIS_URL,
	backend=REDIS_URL,
	include=[],
)

@celery_app.task(name="health.ping")
def ping() -> str:
	return "pong"

@celery_app.task(name="scrapers.run_category")
def run_category(category: str = "parliamentary", retries: int = 2, max_records: int = 10) -> int:
	"""Run full category runner via subprocess; returns exit code"""
	cmd = [
		"python",
		"backend/scripts/run_full_category.py",
		"--category",
		str(category),
		"--retries",
		str(int(retries)),
		"--max-records",
		str(int(max_records)),
	]
	res = subprocess.run(cmd, capture_output=True, text=True)
	return int(res.returncode)

# Optional beat schedule (can be overridden by environment)
celery_app.conf.beat_schedule = {
	"run-parliamentary-daily": {
		"task": "scrapers.run_category",
		"schedule": timedelta(hours=24),
		"args": ("parliamentary", 2, 10),
	},
}