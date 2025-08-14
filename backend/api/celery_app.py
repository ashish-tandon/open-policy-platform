import os
from celery import Celery

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