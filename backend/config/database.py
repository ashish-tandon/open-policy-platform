"""
Database Configuration for Unified Open Policy Platform
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

class DatabaseConfig(BaseSettings):
    """Database configuration settings"""
    
    # Database connection
    host: str = "localhost"
    port: int = 5432
    database: str = "openpolicy"
    username: str = os.getenv("DB_USERNAME", "openpolicy")
    password: str = os.getenv("DB_PASSWORD", "")
    
    # Connection pool settings
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    # SSL settings
    ssl_mode: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_prefix = "DB_"
    
    def get_url(self) -> str:
        """Get database URL"""
        if self.password:
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        else:
            return f"postgresql://{self.username}@{self.host}:{self.port}/{self.database}"
    
    def get_async_url(self) -> str:
        """Get async database URL"""
        base_url = self.get_url()
        return base_url.replace("postgresql://", "postgresql+asyncpg://")

# Global database configuration
db_config = DatabaseConfig()

def create_database_engine() -> Engine:
    """Create database engine"""
    engine = create_engine(
        db_config.get_url(),
        pool_size=db_config.pool_size,
        max_overflow=db_config.max_overflow,
        pool_timeout=db_config.pool_timeout,
        pool_recycle=db_config.pool_recycle,
        echo=False  # Set to True for SQL debugging
    )
    return engine

def get_session_factory():
    """Get session factory"""
    engine = create_database_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_database_session() -> Session:
    """Get database session"""
    SessionLocal = get_session_factory()
    return SessionLocal()

# Global engine instance
engine = create_database_engine()
SessionLocal = get_session_factory()
