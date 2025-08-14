#!/usr/bin/env python3
"""
Initialize scrapers database schema (idempotent).
Creates minimal tables used by monitoring and future writes.
"""
from sqlalchemy import text as sql_text
from backend.config.database import scrapers_engine

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
"""

def main() -> int:
    with scrapers_engine.begin() as conn:
        for stmt in [s for s in DDL.split(";\n") if s.strip()]:
            conn.execute(sql_text(stmt))
    print("Scrapers DB initialized.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())