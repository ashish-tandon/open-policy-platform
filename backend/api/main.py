"""
Main FastAPI Application for Unified Open Policy Platform
"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import uvicorn
from typing import List

# Import all routers
from .routers import policies, scrapers, admin, auth, health, scraper_monitoring, data_management, dashboard

# Import middleware
from .middleware.performance import PerformanceMiddleware
from .middleware.security import SecurityMiddleware, InputValidationMiddleware, RateLimitMiddleware

from .dependencies import get_current_user
from .config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("🚀 Starting Open Policy Platform API...")
    print(f"📊 Database: {settings.database_url}")
    print(f"🔧 Environment: {settings.environment}")
    print("🛡️ Security middleware enabled")
    print("⚡ Performance middleware enabled")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Open Policy Platform API...")

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
    # Back-compat unversioned prefix for health
    app.include_router(health.router, prefix="/api", tags=["Health"])
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
    # Back-compat unversioned prefix
    app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
    app.include_router(policies.router, prefix="/api/v1/policies", tags=["Policies"])
    app.include_router(scrapers.router, prefix="/api/v1/scrapers", tags=["Scrapers"])
    app.include_router(scraper_monitoring.router, tags=["Scraper Monitoring"])
    app.include_router(data_management.router, tags=["Data Management"])
    app.include_router(dashboard.router, tags=["Dashboard"])
    app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])
    
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
