#!/usr/bin/env python3
"""
Initialize scrapers database schema (idempotent).
Creates minimal tables used by monitoring and future writes.
"""
import os
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy import text as sql_text

DDL = """
CREATE TABLE IF NOT EXISTS scraper_results (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    total_scrapers INTEGER NOT NULL,
    successful INTEGER NOT NULL,
    failed INTEGER NOT NULL,
    success_rate DOUBLE PRECISION NOT NULL,
    total_records INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS scraped_representatives (
    id SERIAL PRIMARY KEY,
    jurisdiction TEXT,
    name TEXT NOT NULL,
    role TEXT,
    party TEXT,
    district TEXT,
    email TEXT,
    phone TEXT,
    source TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS scraper_runs (
    id SERIAL PRIMARY KEY,
    category TEXT NOT NULL,
    start_time TIMESTAMPTZ DEFAULT NOW(),
    end_time TIMESTAMPTZ,
    status TEXT NOT NULL,
    records_collected INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS scraper_attempts (
    id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES scraper_runs(id) ON DELETE CASCADE,
    scraper_name TEXT NOT NULL,
    attempt_number INTEGER NOT NULL,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL,
    error_message TEXT
);
"""

def _get_env_url() -> Optional[str]:
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

def _create_engine():
    url = _get_env_url()
    if not url:
        raise RuntimeError("No database URL configured for scrapers DB")
    return create_engine(url, pool_pre_ping=True)

def main() -> int:
    engine = _create_engine()
    with engine.begin() as conn:
        for stmt in [s for s in DDL.split(";\n") if s.strip()]:
            conn.execute(sql_text(stmt))
    print("Scrapers DB initialized.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())