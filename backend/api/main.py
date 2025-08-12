"""
Main FastAPI Application for Unified Open Policy Platform
"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import uvicorn
from typing import List
import logging
import os

# Import all routers
from .routers import policies, scrapers, admin, auth, health, scraper_monitoring, data_management, dashboard
from .routers import metrics as metrics_router

# Import middleware
from .middleware.performance import PerformanceMiddleware
from .middleware.security import SecurityMiddleware, InputValidationMiddleware, RateLimitMiddleware

from .dependencies import get_current_user
from .config import settings

logger = logging.getLogger("openpolicy.api")
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

REQUIRED_ENVS = [
    ("DATABASE_URL", lambda: bool(settings.database_url)),
    ("SECRET_KEY", lambda: bool(settings.secret_key) and settings.secret_key != "your-secret-key-change-in-production"),
]

# Additional production-only policy checks
def _prod_policy_issues() -> list[str]:
    issues: list[str] = []
    if settings.environment.lower() == "production":
        # ALLOWED_HOSTS must not be ["*"] and not empty
        if not settings.allowed_hosts or settings.allowed_hosts == ["*"]:
            issues.append("ALLOWED_HOSTS")
        # ALLOWED_ORIGINS must be set and must not contain localhost/127.0.0.1
        bad = any(("localhost" in o or "127.0.0.1" in o or o == "*") for o in (settings.allowed_origins or []))
        if not settings.allowed_origins or bad:
            issues.append("ALLOWED_ORIGINS")
    return issues

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    missing = []
    for name, checker in REQUIRED_ENVS:
        try:
            if not checker():
                missing.append(name)
        except Exception:
            missing.append(name)
    prod_issues = _prod_policy_issues()
    if missing or prod_issues:
        if missing:
            logger.error("Startup guard failed. Missing/invalid required environment variables: %s", ", ".join(missing))
        if prod_issues:
            logger.error("Startup guard policy violations (production): %s", ", ".join(prod_issues))
        if settings.environment.lower() == "production":
            raise RuntimeError(f"Startup guard failed: missing={missing} policy={prod_issues}")
        else:
            logger.warning("Proceeding in %s with issues: missing=%s policy=%s", settings.environment, missing, prod_issues)

    logger.info("🚀 Starting Open Policy Platform API…")
    logger.info("📊 Database: %s", settings.database_url)
    logger.info("🔧 Environment: %s", settings.environment)
    logger.info("🛡️ Security middleware enabled")
    logger.info("⚡ Performance middleware enabled")
    # Scraper dirs info
    reports_dir = settings.scraper_reports_dir or os.getcwd()
    logs_dir = settings.scraper_logs_dir or os.getcwd()
    app.state.scraper_reports_dir = reports_dir
    app.state.scraper_logs_dir = logs_dir
    logger.info("📁 Scraper reports dir: %s", reports_dir)
    logger.info("📁 Scraper logs dir: %s", logs_dir)
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Open Policy Platform API…")

def create_app() -> FastAPI:
    """Create FastAPI application"""
    
    app = FastAPI(
        title="Open Policy Platform API",
        description="Unified API for policy analysis, data collection, and administration",
        version="1.0.0",
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
        lifespan=lifespan
    )
    
    # Add security middleware first
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(InputValidationMiddleware)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=100)
    
    # Add performance middleware
    app.add_middleware(PerformanceMiddleware, cache_ttl=300, rate_limit_per_minute=100)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts
    )
    
    # Include routers
    app.include_router(health.router, prefix="/api/v1", tags=["Health"])
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
    app.include_router(policies.router, prefix="/api/v1/policies", tags=["Policies"])
    app.include_router(scrapers.router, prefix="/api/v1/scrapers", tags=["Scrapers"])
    app.include_router(scraper_monitoring.router, tags=["Scraper Monitoring"])
    app.include_router(data_management.router, tags=["Data Management"])
    app.include_router(dashboard.router, tags=["Dashboard"])
    app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])
    app.include_router(metrics_router.router, tags=["Metrics"])  # /metrics
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "Open Policy Platform API",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs" if settings.environment != "production" else None,
            "security": "enabled",
            "performance": "optimized"
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": "2024-08-08T00:00:00Z",
            "version": "1.0.0",
            "security": "enabled",
            "performance": "optimized"
        }
    
    return app

# Create app instance
app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower()
    )
