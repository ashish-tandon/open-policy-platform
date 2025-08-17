"""
Pytest configuration and fixtures for OpenPolicy Merge tests
"""

import pytest
import asyncio
from typing import Generator, AsyncGenerator
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import os
# Force tests to use SQLite in-memory DB before importing app
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from api.main import app
from config.database import get_database_session

# Use in-memory SQLite for tests to avoid external dependencies
TEST_DATABASE_URL = "sqlite:///:memory:"

# Create test engine
# For in-memory SQLite shared across threads, use StaticPool and check_same_thread=False
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)

# Create minimal schema required by tests
def _initialize_test_schema():
    with test_engine.connect() as connection:
        # Enable foreign key constraints in SQLite
        connection.execute(text("PRAGMA foreign_keys=ON;"))

        # auth_user table
        connection.execute(text(
            """
            CREATE TABLE IF NOT EXISTS auth_user (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                password TEXT,
                is_active BOOLEAN DEFAULT 1,
                is_staff BOOLEAN DEFAULT 0
            );
            """
        ))

        # users_user table (some tests reference this legacy name)
        connection.execute(text(
            """
            CREATE TABLE IF NOT EXISTS users_user (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                password_hash TEXT,
                first_name TEXT,
                last_name TEXT,
                is_active BOOLEAN DEFAULT 1,
                is_admin BOOLEAN DEFAULT 0
            );
            """
        ))

        # politicians_politician
        connection.execute(text(
            """
            CREATE TABLE IF NOT EXISTS politicians_politician (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                party TEXT,
                constituency TEXT,
                jurisdiction TEXT,
                email TEXT,
                phone TEXT,
                updated_2025 BOOLEAN DEFAULT 0,
                data_source_2025 TEXT,
                last_modified TEXT
            );
            """
        ))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_politicians_name ON politicians_politician(name);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_politicians_jurisdiction ON politicians_politician(jurisdiction);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_politicians_party ON politicians_politician(party);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_politicians_constituency ON politicians_politician(constituency);"))

        # bills_bill (superset of columns referenced by tests)
        connection.execute(text(
            """
            CREATE TABLE IF NOT EXISTS bills_bill (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                description TEXT,
                bill_number TEXT UNIQUE,
                introduced_date TEXT,
                sponsor TEXT,
                jurisdiction TEXT,
                status TEXT,
                updated_2025 BOOLEAN DEFAULT 0,
                data_source_2025 TEXT,
                last_modified TEXT,
                -- legacy columns referenced in some tests
                name_en TEXT,
                name_fr TEXT,
                number TEXT,
                number_only INTEGER,
                institution TEXT,
                status_code TEXT,
                added TEXT,
                session_id TEXT,
                library_summary_available BOOLEAN,
                short_title_en TEXT,
                short_title_fr TEXT
            );
            """
        ))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_bills_bill_number ON bills_bill(bill_number);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_bills_jurisdiction ON bills_bill(jurisdiction);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_bills_introduced_date ON bills_bill(introduced_date);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_bills_sponsor ON bills_bill(sponsor);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_bills_juris_date ON bills_bill(jurisdiction, introduced_date);"))

        # votes_vote
        connection.execute(text(
            """
            CREATE TABLE IF NOT EXISTS votes_vote (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bill_number TEXT,
                vote_date TEXT,
                vote_type TEXT,
                result TEXT,
                yea_votes INTEGER,
                nay_votes INTEGER,
                abstentions INTEGER,
                jurisdiction TEXT,
                updated_2025 BOOLEAN DEFAULT 0,
                data_source_2025 TEXT,
                last_modified TEXT,
                FOREIGN KEY (bill_number) REFERENCES bills_bill(bill_number)
            );
            """
        ))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_votes_bill_number ON votes_vote(bill_number);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_votes_vote_date ON votes_vote(vote_date);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_votes_jurisdiction ON votes_vote(jurisdiction);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_votes_result ON votes_vote(result);"))

        # committees_committee
        connection.execute(text(
            """
            CREATE TABLE IF NOT EXISTS committees_committee (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                description TEXT,
                jurisdiction TEXT,
                members INTEGER,
                updated_2025 BOOLEAN DEFAULT 0,
                data_source_2025 TEXT,
                last_modified TEXT,
                FOREIGN KEY (members) REFERENCES politicians_politician(id)
            );
            """
        ))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_committees_name ON committees_committee(name);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_committees_jurisdiction ON committees_committee(jurisdiction);"))

        # hansards_statement
        connection.execute(text(
            """
            CREATE TABLE IF NOT EXISTS hansards_statement (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                speaker TEXT,
                content TEXT,
                date TEXT,
                session TEXT,
                jurisdiction TEXT,
                updated_2025 BOOLEAN DEFAULT 0,
                data_source_2025 TEXT,
                last_modified TEXT
            );
            """
        ))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_hansards_speaker ON hansards_statement(speaker);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_hansards_date ON hansards_statement(date);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS idx_hansards_jurisdiction ON hansards_statement(jurisdiction);"))

        # activity_activity (minimal stub)
        connection.execute(text(
            """
            CREATE TABLE IF NOT EXISTS activity_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT
            );
            """
        ))

        # alerts_subscription (minimal stub)
        connection.execute(text(
            """
            CREATE TABLE IF NOT EXISTS alerts_subscription (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT
            );
            """
        ))

        # django tables (minimal stubs)
        connection.execute(text(
            """
            CREATE TABLE IF NOT EXISTS django_content_type (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                app_label TEXT,
                model TEXT
            );
            """
        ))
        connection.execute(text(
            """
            CREATE TABLE IF NOT EXISTS django_migrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                app TEXT,
                name TEXT,
                applied TEXT
            );
            """
        ))
        connection.execute(text(
            """
            CREATE TABLE IF NOT EXISTS django_session (
                session_key TEXT PRIMARY KEY,
                session_data TEXT,
                expire_date TEXT
            );
            """
        ))

# Initialize schema once at import time
_initialize_test_schema()

# Create test session factory
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def db_session():
    """Create a fresh database session for each test."""
    connection = test_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture
def client(db_session) -> Generator:
    """Create a test client with database session."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_database_session] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture
def auth_headers():
    """Get authentication headers for admin user."""
    return {
        "Authorization": "Bearer test_admin_token"
    }

@pytest.fixture
def sample_policy_data():
    """Sample policy data for testing."""
    return {
        "title": "Test Policy",
        "description": "A test policy for testing purposes",
        "jurisdiction": "federal",
        "status": "active",
        "introduced_date": "2024-01-01",
        "sponsor": "Test Sponsor"
    }

@pytest.fixture
def sample_representative_data():
    """Sample representative data for testing."""
    return {
        "name": "Test Representative",
        "party": "Test Party",
        "jurisdiction": "federal",
        "constituency": "Test Constituency",
        "email": "test@example.com",
        "phone": "123-456-7890"
    }

@pytest.fixture
def sample_scraper_data():
    """Sample scraper data for testing."""
    return {
        "name": "test_scraper",
        "jurisdiction": "federal",
        "status": "active",
        "last_run": "2024-01-01T00:00:00Z",
        "records_scraped": 100
    }
