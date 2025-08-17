# 🚀 Backend API - Main Application Entry Point

## 📄 **FILE OVERVIEW**

**File Path**: `backend/api/main.py`  
**Purpose**: Main FastAPI application entry point and configuration  
**Technology**: FastAPI with comprehensive middleware and router integration  
**Last Updated**: January 16, 2025  

---

## 🎯 **COMPONENT PURPOSE**

The `main.py` file serves as the central entry point for the Open Policy Platform API. It orchestrates the entire application lifecycle, configures middleware, integrates all API routers, and manages application startup/shutdown procedures.

---

## 🏗️ **ARCHITECTURE ROLE**

### **System Position**
```
User Request → API Gateway → main.py → Router → Service → Database
```

### **Responsibilities**
- **Application Lifecycle**: Startup, shutdown, and health management
- **Middleware Configuration**: Security, performance, and monitoring
- **Router Integration**: All API endpoint registration
- **Configuration Management**: Environment and settings validation
- **Error Handling**: Global exception handling and logging
- **Health Monitoring**: System status and dependency checks

---

## 📁 **FILE STRUCTURE**

```python
# File: backend/api/main.py
"""
Main FastAPI Application for Unified Open Policy Platform
"""

# 1. IMPORTS SECTION
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import uvicorn
from typing import List
import logging
import os

# 2. ROUTER IMPORTS
from .routers import policies, scrapers, admin, auth, health, scraper_monitoring, data_management, dashboard
from .routers import metrics as metrics_router
from .routers import representatives, committees, debates, votes, search, analytics, notifications, files

# 3. MIDDLEWARE IMPORTS
from .middleware.performance import PerformanceMiddleware
from .middleware.security import SecurityMiddleware, InputValidationMiddleware, RateLimitMiddleware

# 4. DEPENDENCIES AND CONFIG
from .dependencies import get_current_user
from .config import settings

# 5. APPLICATION CONFIGURATION
logger = logging.getLogger("openpolicy.api")
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

# 6. ENVIRONMENT VALIDATION
REQUIRED_ENVS = [
    ("DATABASE_URL", lambda: bool(settings.database_url)),
    ("SECRET_KEY", lambda: bool(settings.secret_key) and settings.secret_key != "your-secret-key-change-in-production"),
]

# 7. PRODUCTION POLICY CHECKS
def _prod_policy_issues() -> list[str]:
    # Production environment validation logic

# 8. LIFECYCLE MANAGER
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Application startup and shutdown logic

# 9. APPLICATION FACTORY
def create_app() -> FastAPI:
    # FastAPI application creation and configuration

# 10. APPLICATION INSTANCE
app = create_app()

# 11. MAIN EXECUTION
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 🔧 **CORE COMPONENTS**

### **1. Import Management**
**Purpose**: Centralized import organization for all system components

#### **Router Imports**
```python
# Core API Routers
from .routers import policies, scrapers, admin, auth, health, scraper_monitoring, data_management, dashboard

# Enhanced Microservices Routers (from v2-cursor)
from .routers import representatives, committees, debates, votes, search, analytics, notifications, files

# Metrics Router
from .routers import metrics as metrics_router
```

#### **Middleware Imports**
```python
# Performance and Security Middleware
from .middleware.performance import PerformanceMiddleware
from .middleware.security import SecurityMiddleware, InputValidationMiddleware, RateLimitMiddleware
```

#### **Configuration Imports**
```python
# Dependencies and Settings
from .dependencies import get_current_user
from .config import settings
```

### **2. Environment Validation**
**Purpose**: Ensure required environment variables are properly configured

#### **Required Environment Variables**
```python
REQUIRED_ENVS = [
    ("DATABASE_URL", lambda: bool(settings.database_url)),
    ("SECRET_KEY", lambda: bool(settings.secret_key) and settings.secret_key != "your-secret-key-change-in-production"),
]
```

#### **Production Policy Validation**
```python
def _prod_policy_issues() -> list[str]:
    issues: list[str] = []
    if settings.environment.lower() == "production":
        # ALLOWED_HOSTS validation
        if not settings.allowed_hosts or settings.allowed_hosts == ["*"]:
            issues.append("ALLOWED_HOSTS")
        # ALLOWED_ORIGINS validation
        bad = any(("localhost" in o or "127.0.0.1" in o or o == "*") for o in (settings.allowed_origins or []))
        if not settings.allowed_origins or bad:
            issues.append("ALLOWED_ORIGINS")
    return issues
```

### **3. Application Lifecycle Management**
**Purpose**: Manage application startup, shutdown, and health monitoring

#### **Lifespan Manager**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    
    # STARTUP PHASE
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
    
    # Startup logging and configuration
    logger.info("🚀 Starting Open Policy Platform API…")
    logger.info("📊 Database: %s", settings.database_url)
    logger.info("🔧 Environment: %s", settings.environment)
    logger.info("🛡️ Security middleware enabled")
    logger.info("⚡ Performance middleware enabled")
    
    # Scraper directory configuration
    reports_dir = settings.scraper_reports_dir or os.getcwd()
    logs_dir = settings.scraper_logs_dir or os.getcwd()
    app.state.scraper_reports_dir = reports_dir
    app.state.scraper_logs_dir = logs_dir
    logger.info("📁 Scraper reports dir: %s", reports_dir)
    logger.info("📁 Scraper logs dir: %s", logs_dir)
    
    yield  # Application runs here
    
    # SHUTDOWN PHASE
    logger.info("🛑 Shutting down Open Policy Platform API…")
```

### **4. Application Factory**
**Purpose**: Create and configure the FastAPI application instance

#### **Application Creation**
```python
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
    
    # Middleware configuration
    app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    app.add_middleware(PerformanceMiddleware)
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(InputValidationMiddleware)
    app.add_middleware(RateLimitMiddleware)
    
    # Router registration
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
    app.include_router(policies.router, prefix="/api/v1/policies", tags=["Policies"])
    app.include_router(scrapers.router, prefix="/api/v1/scrapers", tags=["Scrapers"])
    app.include_router(admin.router, prefix="/api/v1/admin", tags=["Administration"])
    app.include_router(health.router, prefix="/api/v1/health", tags=["Health"])
    app.include_router(metrics_router.router, prefix="/api/v1/metrics", tags=["Metrics"])
    app.include_router(scraper_monitoring.router, prefix="/api/v1/scrapers", tags=["Scraper Monitoring"])
    app.include_router(data_management.router, prefix="/api/v1/data", tags=["Data Management"])
    app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["Dashboard"])
    
    # Enhanced microservices routers
    app.include_router(representatives.router, prefix="/api/v1/representatives", tags=["Representatives"])
    app.include_router(committees.router, prefix="/api/v1/committees", tags=["Committees"])
    app.include_router(debates.router, prefix="/api/v1/debates", tags=["Debates"])
    app.include_router(votes.router, prefix="/api/v1/votes", tags=["Votes"])
    app.include_router(search.router, prefix="/api/v1/search", tags=["Search"])
    app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])
    app.include_router(notifications.router, prefix="/api/v1/notifications", tags=["Notifications"])
    app.include_router(files.router, prefix="/api/v1/files", tags=["Files"])
    
    return app
```

---

## 🔄 **DATA FLOW ARCHITECTURE**

### **Request Processing Flow**
```
HTTP Request → FastAPI App → Middleware Stack → Router → Handler → Response
```

### **Middleware Execution Order**
1. **CORS Middleware**: Cross-origin request handling
2. **Trusted Host Middleware**: Host validation
3. **Performance Middleware**: Request timing and metrics
4. **Security Middleware**: Security headers and validation
5. **Input Validation Middleware**: Request data validation
6. **Rate Limit Middleware**: Request rate limiting

### **Router Execution Flow**
```
Request → Router Selection → Authentication → Authorization → Handler → Response
```

---

## 🛡️ **SECURITY FEATURES**

### **Production Environment Guards**
- **Environment Variable Validation**: Required variables must be present
- **Security Policy Enforcement**: Production-specific security requirements
- **Host Validation**: Trusted host configuration
- **CORS Configuration**: Cross-origin request control

### **Security Middleware Stack**
- **Security Headers**: Security-related HTTP headers
- **Input Validation**: Request data sanitization
- **Rate Limiting**: Request frequency control
- **Authentication**: JWT token validation

---

## 📊 **MONITORING & OBSERVABILITY**

### **Health Monitoring**
- **Startup Health Checks**: Environment and configuration validation
- **Runtime Health**: Service availability monitoring
- **Shutdown Grace**: Graceful service termination

### **Logging Configuration**
- **Structured Logging**: JSON-formatted log entries
- **Log Levels**: Configurable logging verbosity
- **Startup Logging**: Comprehensive startup information
- **Error Logging**: Detailed error tracking

### **Metrics Collection**
- **Performance Metrics**: Request timing and throughput
- **Health Metrics**: Service health status
- **Business Metrics**: API usage and performance

---

## 🔧 **CONFIGURATION MANAGEMENT**

### **Environment Configuration**
- **Database Configuration**: Connection string validation
- **Security Configuration**: Secret key and security settings
- **Scraper Configuration**: Directory and path settings
- **API Configuration**: Server and endpoint settings

### **Dynamic Configuration**
- **Environment-based Settings**: Development vs production
- **Runtime Configuration**: Dynamic configuration updates
- **Health Check Configuration**: Configurable health monitoring

---

## 🚀 **DEPLOYMENT CONSIDERATIONS**

### **Container Configuration**
- **Port Configuration**: Default port 8000
- **Host Configuration**: Bind to all interfaces (0.0.0.0)
- **Environment Variables**: Configuration via environment
- **Health Checks**: Container health monitoring

### **Scaling Considerations**
- **Stateless Design**: No local state storage
- **Horizontal Scaling**: Multiple instance support
- **Load Balancing**: External load balancer support
- **Health Monitoring**: External health check endpoints

---

## 🧪 **TESTING STRATEGY**

### **Unit Testing**
- **Component Testing**: Individual function testing
- **Mock Testing**: External dependency mocking
- **Configuration Testing**: Environment validation testing

### **Integration Testing**
- **API Testing**: End-to-end API testing
- **Middleware Testing**: Middleware stack testing
- **Router Testing**: Router integration testing

### **Performance Testing**
- **Load Testing**: High-volume request testing
- **Stress Testing**: Resource exhaustion testing
- **Memory Testing**: Memory leak detection

---

## 🔍 **TROUBLESHOOTING GUIDE**

### **Common Issues**

#### **1. Environment Variable Errors**
```bash
# Error: Startup guard failed. Missing/invalid required environment variables
# Solution: Ensure DATABASE_URL and SECRET_KEY are set
export DATABASE_URL="postgresql://user:pass@localhost/db"
export SECRET_KEY="your-secure-secret-key"
```

#### **2. Production Policy Violations**
```bash
# Error: Startup guard policy violations (production)
# Solution: Configure proper ALLOWED_HOSTS and ALLOWED_ORIGINS
export ALLOWED_HOSTS=["yourdomain.com"]
export ALLOWED_ORIGINS=["https://yourdomain.com"]
```

#### **3. Scraper Directory Issues**
```bash
# Error: Scraper directories not accessible
# Solution: Ensure directories exist and are writable
mkdir -p /app/scraper-reports /app/scraper-logs
chmod 755 /app/scraper-reports /app/scraper-logs
```

### **Debugging Commands**
```bash
# Check environment variables
env | grep -E "(DATABASE_URL|SECRET_KEY|ENVIRONMENT)"

# Check scraper directories
ls -la /app/scraper-*

# Check application logs
tail -f /app/logs/app.log

# Test health endpoints
curl http://localhost:8000/api/v1/health
```

---

## 📚 **REFERENCE MATERIALS**

### **FastAPI Documentation**
- **Official Docs**: https://fastapi.tiangolo.com/
- **Middleware Guide**: https://fastapi.tiangolo.com/tutorial/middleware/
- **Lifespan Events**: https://fastapi.tiangolo.com/advanced/events/

### **Configuration Reference**
- **Environment Variables**: Complete list of required variables
- **Settings Classes**: Configuration class definitions
- **Validation Rules**: Environment validation requirements

### **API Endpoints**
- **Health Endpoints**: `/api/v1/health`
- **Metrics Endpoints**: `/api/v1/metrics`
- **Documentation**: `/docs` (Swagger UI)

---

## 🎯 **DEVELOPMENT WORKFLOW**

### **Adding New Routers**
1. **Create Router File**: Add new router in `routers/` directory
2. **Import Router**: Add import to main.py
3. **Register Router**: Include router with appropriate prefix and tags
4. **Update Documentation**: Document new endpoints and functionality

### **Adding New Middleware**
1. **Create Middleware**: Implement middleware class
2. **Import Middleware**: Add import to main.py
3. **Register Middleware**: Add to middleware stack in correct order
4. **Test Middleware**: Verify middleware execution order

### **Configuration Changes**
1. **Update Settings**: Modify configuration classes
2. **Add Validation**: Add environment variable validation
3. **Update Documentation**: Document new configuration options
4. **Test Configuration**: Verify configuration loading

---

## 🔄 **VERSION HISTORY**

### **v1.0.0 (Current)**
- **Initial Implementation**: Basic FastAPI application structure
- **Core Routers**: Authentication, policies, scrapers, admin
- **Basic Middleware**: CORS, security, performance
- **Health Monitoring**: Basic health check endpoints

### **v2.0.0 (v2-cursor Integration)**
- **Enhanced Routers**: Analytics, committees, debates, votes, search, notifications, files
- **Advanced Middleware**: Input validation, rate limiting
- **Comprehensive Monitoring**: Enhanced health checks and metrics
- **Production Readiness**: Environment validation and security policies

---

## 🎯 **NEXT STEPS**

### **Immediate Actions**
1. **Router Documentation**: Document each individual router
2. **Middleware Documentation**: Document middleware components
3. **Configuration Documentation**: Document all configuration options
4. **Testing Documentation**: Document testing procedures

### **Future Enhancements**
1. **Advanced Monitoring**: Distributed tracing and metrics
2. **Configuration Management**: Dynamic configuration updates
3. **Performance Optimization**: Request caching and optimization
4. **Security Enhancements**: Advanced security features

---

**🎯 This documentation provides comprehensive understanding of the main.py file. It serves as the foundation for understanding how the entire API system is orchestrated and configured.**
