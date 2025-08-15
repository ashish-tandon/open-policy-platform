FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	libxml2-dev libxslt1-dev zlib1g-dev \
	curl ca-certificates git \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /src/scrapers-ca

COPY ../requirements.in ./requirements.in
RUN pip install --no-cache-dir -r requirements.in || true

COPY .. /src/scrapers-ca

CMD ["python", "-c", "print('scrapers-ca container ready'); import time; time.sleep(3600)"]
