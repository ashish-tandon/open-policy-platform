# 📋 Service Documentation Template - Open Policy Platform

## 🎯 **TEMPLATE OVERVIEW**

> **"Every service must be fully documented with complete dependency mapping, testing, deployment, and monitoring information"**

This template defines the mandatory documentation structure that every service in the Open Policy Platform must follow. Use this template to ensure comprehensive service documentation compliance.

---

## 📚 **MANDATORY DOCUMENTATION SECTIONS**

### **1. SERVICE OVERVIEW** (Required)
```markdown
## 🎯 **SERVICE OVERVIEW**

### **Service Identification**
- **Service Name**: [Service Name]
- **Service Version**: [Current Version]
- **Service Type**: [API Service, Background Service, Infrastructure Service, etc.]
- **Repository Path**: [Path to service in repository]

### **Purpose and Functionality**
- **Primary Purpose**: [Clear description of what this service does]
- **Key Features**: [List of main features and capabilities]
- **Business Value**: [How this service contributes to business objectives]

### **Architecture Role**
- **Position in Architecture**: [Where this service fits in the overall system]
- **Responsibility Level**: [Primary, Secondary, Support, Infrastructure]
- **Integration Points**: [How this service integrates with other components]

### **Technology Stack**
- **Programming Language**: [Python, Go, JavaScript, etc.]
- **Framework**: [FastAPI, Django, Express, etc.]
- **Database**: [PostgreSQL, MongoDB, Redis, etc.]
- **Dependencies**: [Key external libraries and tools]
```

### **2. SERVICE DEPENDENCIES** (Required)
```markdown
## 🔗 **SERVICE DEPENDENCIES**

### **Required Services** (Must be available for service to function)
| Service | Purpose | Port | Protocol | Health Check |
|---------|---------|------|----------|--------------|
| [Service Name] | [Purpose] | [Port] | [Protocol] | [Health Endpoint] |
| [Service Name] | [Purpose] | [Port] | [Protocol] | [Health Endpoint] |

### **Optional Services** (Service can function without, but with reduced capability)
| Service | Purpose | Port | Protocol | Health Check | Fallback Behavior |
|---------|---------|------|----------|--------------|-------------------|
| [Service Name] | [Purpose] | [Port] | [Protocol] | [Health Endpoint] | [What happens if unavailable] |

### **External Dependencies** (Third-party services and APIs)
| Service | Purpose | Endpoint | Authentication | Rate Limits | Health Check |
|---------|---------|----------|----------------|-------------|--------------|
| [Service Name] | [Purpose] | [URL] | [Auth Method] | [Limits] | [Health Check] |

### **Database Dependencies**
| Database | Purpose | Schema | Connection Pool | Health Check |
|----------|---------|--------|----------------|--------------|
| [Database Name] | [Purpose] | [Schema Name] | [Pool Size] | [Health Endpoint] |

### **Dependency Health Monitoring**
```yaml
dependency_health:
  required_services:
    - service: database
      health_endpoint: "/health/database"
      timeout: 5s
      retry_count: 3
      critical: true
    
    - service: redis
      health_endpoint: "/health/cache"
      timeout: 2s
      retry_count: 2
      critical: false
```
```

### **3. SERVICE CONFIGURATION** (Required)
```markdown
## ⚙️ **SERVICE CONFIGURATION**

### **Environment Variables**
| Variable | Purpose | Required | Default | Validation |
|----------|---------|----------|---------|------------|
| `DATABASE_URL` | Database connection string | Yes | None | Must be valid PostgreSQL URL |
| `SECRET_KEY` | Service secret key | Yes | None | Must be 32+ characters |
| `LOG_LEVEL` | Logging verbosity | No | INFO | DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `PORT` | Service port | No | 8000 | Must be 1024-65535 |

### **Configuration Files**
| File | Purpose | Format | Location | Required |
|------|---------|--------|----------|----------|
| `config.py` | Service configuration | Python | `./config/` | Yes |
| `.env` | Environment variables | Key-Value | `./` | Yes |
| `logging.conf` | Logging configuration | INI | `./config/` | No |

### **Configuration Validation**
```python
# Required configuration validation
def validate_configuration():
    required_vars = ["DATABASE_URL", "SECRET_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ConfigurationError(f"Missing required environment variables: {missing_vars}")
    
    # Validate specific configurations
    validate_database_url()
    validate_secret_key()
    validate_port_configuration()
```

### **Default Values and Overrides**
```yaml
defaults:
  port: 8000
  log_level: INFO
  database_pool_size: 10
  cache_ttl: 3600

overrides:
  development:
    log_level: DEBUG
    database_pool_size: 5
  
  production:
    log_level: WARNING
    database_pool_size: 20
```
```

### **4. SERVICE ENDPOINTS** (Required)
```markdown
## 🌐 **SERVICE ENDPOINTS**

### **API Endpoints**
| Endpoint | Method | Purpose | Authentication | Rate Limit | Response Format |
|----------|--------|---------|----------------|------------|-----------------|
| `/api/v1/resource` | GET | Retrieve resource | JWT Required | 100/min | JSON |
| `/api/v1/resource` | POST | Create resource | JWT Required | 50/min | JSON |
| `/api/v1/resource/{id}` | PUT | Update resource | JWT Required | 50/min | JSON |
| `/api/v1/resource/{id}` | DELETE | Delete resource | JWT Required | 25/min | JSON |

### **Health Check Endpoints**
| Endpoint | Method | Purpose | Response | Health Status |
|----------|--------|---------|----------|---------------|
| `/health` | GET | Basic health check | `{"status": "healthy"}` | Service health |
| `/health/database` | GET | Database health | `{"status": "healthy", "database": "connected"}` | Database connectivity |
| `/health/dependencies` | GET | Dependency health | `{"status": "healthy", "dependencies": {...}}` | All dependencies |

### **Metrics Endpoints**
| Endpoint | Method | Purpose | Response | Format |
|----------|--------|---------|----------|--------|
| `/metrics` | GET | Prometheus metrics | Metrics data | Prometheus format |
| `/metrics/performance` | GET | Performance metrics | Performance data | JSON |

### **Request/Response Examples**
```json
// Example Request
{
  "resource_name": "example",
  "description": "Example resource",
  "metadata": {
    "tags": ["example", "test"]
  }
}

// Example Response
{
  "id": "uuid-here",
  "resource_name": "example",
  "description": "Example resource",
  "metadata": {
    "tags": ["example", "test"]
  },
  "created_at": "2025-01-16T10:30:00Z",
  "updated_at": "2025-01-16T10:30:00Z"
}
```

### **Error Response Format**
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "resource_name",
      "issue": "Field is required"
    },
    "timestamp": "2025-01-16T10:30:00Z",
    "request_id": "uuid-here"
  }
}
```
```

### **5. SERVICE PORTS AND CONNECTIVITY** (Required)
```markdown
## 🔌 **SERVICE PORTS AND CONNECTIVITY**

### **Service Ports**
| Port | Purpose | Protocol | Access | Health Check |
|------|---------|----------|--------|--------------|
| 8000 | HTTP API | HTTP | External | `/health` |
| 8001 | Admin API | HTTP | Internal | `/admin/health` |
| 5432 | Database | PostgreSQL | Internal | Database connection |

### **Internal Communication**
| Service | Purpose | Port | Protocol | Authentication | Health Check |
|---------|---------|------|----------|----------------|--------------|
| [Service Name] | [Purpose] | [Port] | [Protocol] | [Auth Method] | [Health Endpoint] |
| [Service Name] | [Purpose] | [Port] | [Protocol] | [Auth Method] | [Health Endpoint] |

### **External Communication**
| Service | Purpose | Endpoint | Protocol | Authentication | Rate Limiting |
|---------|---------|----------|----------|----------------|---------------|
| [Service Name] | [Purpose] | [URL] | [Protocol] | [Auth Method] | [Limits] |

### **Network Requirements**
```yaml
network_configuration:
  inbound:
    - port: 8000
      protocol: http
      access: external
      security: tls_required
    
    - port: 8001
      protocol: http
      access: internal
      security: jwt_required
  
  outbound:
    - service: database
      port: 5432
      protocol: postgresql
      security: internal_network
    
    - service: redis
      port: 6379
      protocol: redis
      security: internal_network
```

### **Firewall and Security**
- **Inbound Rules**: [List of allowed inbound connections]
- **Outbound Rules**: [List of allowed outbound connections]
- **Security Groups**: [Security group configurations]
- **Network Policies**: [Kubernetes network policies]
```

### **6. SERVICE TESTING** (Required)
```markdown
## 🧪 **SERVICE TESTING**

### **Unit Tests**
| Test Category | Coverage Target | Test Files | Description |
|---------------|----------------|------------|-------------|
| Core Logic | 90% | `test_core.py` | Business logic testing |
| API Endpoints | 95% | `test_api.py` | Endpoint functionality |
| Configuration | 100% | `test_config.py` | Configuration validation |
| Utilities | 85% | `test_utils.py` | Helper function testing |

### **Integration Tests**
| Test Category | Test Files | Description | Dependencies |
|---------------|------------|-------------|--------------|
| Database | `test_database.py` | Database integration | PostgreSQL |
| External APIs | `test_external.py` | External service integration | Mock services |
| Authentication | `test_auth.py` | Auth system integration | JWT service |

### **Smoke Tests**
```bash
#!/bin/bash
# Required smoke test script

echo "Running smoke tests for [Service Name]..."

# Test 1: Service startup
echo "Test 1: Service startup"
curl -f http://localhost:8000/health || exit 1

# Test 2: Basic functionality
echo "Test 2: Basic functionality"
curl -f http://localhost:8000/api/v1/basic-endpoint || exit 1

# Test 3: Database connection
echo "Test 3: Database connection"
curl -f http://localhost:8000/health/database || exit 1

# Test 4: Dependencies
echo "Test 4: Dependencies"
curl -f http://localhost:8000/health/dependencies || exit 1

echo "All smoke tests passed!"
```

### **Performance Tests**
| Test Type | Test Files | Metrics | Thresholds |
|-----------|------------|---------|------------|
| Load Testing | `test_load.py` | Response time, throughput | < 200ms, > 1000 req/s |
| Stress Testing | `test_stress.py` | Error rate, resource usage | < 5% errors, < 80% CPU |
| Memory Testing | `test_memory.py` | Memory usage, leaks | < 512MB, no leaks |

### **Test Coverage Requirements**
- **Overall Coverage**: Minimum 80%
- **Critical Paths**: Minimum 95%
- **API Endpoints**: Minimum 90%
- **Error Handling**: Minimum 85%

### **Test Execution**
```bash
# Run all tests
pytest tests/ --cov=src --cov-report=html

# Run specific test categories
pytest tests/test_api.py -v
pytest tests/test_database.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing
```
```

### **7. SERVICE DEPLOYMENT** (Required)
```markdown
## 🚀 **SERVICE DEPLOYMENT**

### **Prerequisites**
- [ ] Database service running and accessible
- [ ] Redis service running and accessible
- [ ] Environment variables configured
- [ ] Configuration files in place
- [ ] Dependencies installed

### **Deployment Process**
```yaml
deployment_steps:
  - step: "Environment Setup"
    description: "Configure environment variables"
    commands:
      - "cp .env.example .env"
      - "edit .env with proper values"
    validation: "Verify all required variables are set"
  
  - step: "Dependencies Installation"
    description: "Install service dependencies"
    commands:
      - "pip install -r requirements.txt"
    validation: "Verify all packages installed successfully"
  
  - step: "Database Migration"
    description: "Run database migrations"
    commands:
      - "alembic upgrade head"
    validation: "Verify database schema is current"
  
  - step: "Configuration Validation"
    description: "Validate service configuration"
    commands:
      - "python -c 'from config import validate_config; validate_config()'"
    validation: "Configuration validation passes"
  
  - step: "Service Startup"
    description: "Start the service"
    commands:
      - "python main.py"
    validation: "Service starts without errors"
  
  - step: "Health Verification"
    description: "Verify service health"
    commands:
      - "curl http://localhost:8000/health"
    validation: "Health check returns healthy status"
```

### **Environment-Specific Configurations**
```yaml
environments:
  development:
    log_level: DEBUG
    database_pool_size: 5
    cache_ttl: 300
    
  staging:
    log_level: INFO
    database_pool_size: 10
    cache_ttl: 1800
    
  production:
    log_level: WARNING
    database_pool_size: 20
    cache_ttl: 3600
```

### **Rollback Procedures**
```yaml
rollback_steps:
  - step: "Stop Current Service"
    command: "docker compose stop service-name"
    
  - step: "Revert to Previous Version"
    command: "git checkout previous-version"
    
  - step: "Restart Service"
    command: "docker compose up -d service-name"
    
  - step: "Verify Rollback"
    command: "curl http://localhost:8000/health"
```

### **Health Check Requirements**
```yaml
health_checks:
  startup:
    - endpoint: "/health"
      method: "GET"
      expected_status: 200
      timeout: 30s
      retry_count: 3
  
  runtime:
    - endpoint: "/health"
      method: "GET"
      frequency: "30s"
      expected_status: 200
      timeout: 5s
    
    - endpoint: "/health/database"
      method: "GET"
      frequency: "60s"
      expected_status: 200
      timeout: 10s
```
```

### **8. SERVICE MONITORING** (Required)
```markdown
## 📊 **SERVICE MONITORING**

### **Health Monitoring**
| Endpoint | Purpose | Frequency | Expected Response | Alert Threshold |
|----------|---------|-----------|-------------------|----------------|
| `/health` | Basic health | 30s | `{"status": "healthy"}` | Unhealthy for > 2 checks |
| `/health/database` | Database health | 60s | `{"status": "healthy"}` | Unhealthy for > 1 check |
| `/health/dependencies` | Dependencies | 120s | `{"status": "healthy"}` | Unhealthy for > 1 check |

### **Performance Monitoring**
| Metric | Collection Method | Alert Threshold | Dashboard |
|--------|------------------|-----------------|-----------|
| Response Time | Prometheus | > 500ms | Performance Dashboard |
| Error Rate | Prometheus | > 5% | Error Dashboard |
| Throughput | Prometheus | < 100 req/s | Performance Dashboard |
| Memory Usage | Prometheus | > 80% | Resource Dashboard |

### **Logging Configuration**
```yaml
logging_configuration:
  format: "json"
  level: "INFO"
  handlers:
    - type: "file"
      filename: "logs/services/service-name.log"
      max_bytes: 10485760  # 10MB
      backup_count: 5
    
    - type: "console"
      level: "INFO"
  
  structured_fields:
    - service_name
    - service_version
    - instance_id
    - request_id
    - user_id
    - operation
    - duration_ms
    - status
```

### **Alerting Rules**
```yaml
alerting_rules:
  - name: "Service Unhealthy"
    condition: "health_check_status != 'healthy'"
    duration: "1m"
    severity: "critical"
    notification: "slack,email"
    
  - name: "High Error Rate"
    condition: "error_rate > 5%"
    duration: "5m"
    severity: "warning"
    notification: "slack"
    
  - name: "High Response Time"
    condition: "response_time > 500ms"
    duration: "5m"
    severity: "warning"
    notification: "slack"
```

### **Dashboard Configuration**
| Dashboard | Purpose | Key Metrics | Refresh Rate |
|-----------|---------|-------------|--------------|
| Service Overview | General service status | Health, errors, performance | 30s |
| Performance | Response times, throughput | Response time, requests/sec | 15s |
| Resources | CPU, memory, disk usage | CPU%, memory%, disk% | 30s |
| Errors | Error rates and types | Error count, error types | 15s |
```

### **9. TROUBLESHOOTING** (Required)
```markdown
## 🔍 **TROUBLESHOOTING**

### **Common Issues and Solutions**
| Issue | Symptoms | Root Cause | Solution |
|-------|----------|------------|----------|
| Database Connection Failed | Health check fails | Database service down | Restart database service |
| High Memory Usage | Service slow, memory alerts | Memory leak or high load | Check memory usage, restart if needed |
| API Timeouts | Slow responses, timeouts | External service slow | Check external service health |

### **Debug Procedures**
```bash
# 1. Check service status
docker compose ps service-name

# 2. Check service logs
docker compose logs -f service-name

# 3. Check health endpoints
curl -v http://localhost:8000/health
curl -v http://localhost:8000/health/database

# 4. Check resource usage
docker stats service-name

# 5. Check configuration
docker compose exec service-name env | grep -E "(DATABASE|SECRET|PORT)"
```

### **Log Analysis Commands**
```bash
# Search for errors
grep "ERROR" logs/services/service-name.log

# Search for specific operation
grep "operation_name" logs/services/service-name.log

# Search for performance issues
grep "duration_ms" logs/services/service-name.log | awk '$NF > 1000'

# Search for specific user
grep "user_id:123" logs/services/service-name.log
```

### **Support Contacts**
| Role | Contact | Availability | Escalation |
|------|---------|--------------|------------|
| Primary Developer | [Name] | [Hours] | [Escalation Path] |
| DevOps Engineer | [Name] | [Hours] | [Escalation Path] |
| System Administrator | [Name] | [Hours] | [Escalation Path] |
| Emergency Contact | [Name] | 24/7 | [Escalation Path] |
```

---

## 🔍 **COMPLIANCE CHECKLIST**

### **Documentation Compliance**
- [ ] Service overview complete and accurate
- [ ] All dependencies documented with ports and protocols
- [ ] Configuration requirements fully documented
- [ ] All endpoints documented with examples
- [ ] Ports and connectivity clearly defined
- [ ] Testing procedures documented and implemented
- [ ] Deployment process documented and tested
- [ ] Health checks implemented and documented
- [ ] Logging configured according to standards
- [ ] Monitoring and alerting configured

### **Architecture Compliance**
- [ ] Service follows unified architecture principles
- [ ] Service integrates with centralized logging
- [ ] Service follows dependency management standards
- [ ] Service implements required health checks
- [ ] Service follows security requirements
- [ ] Service follows performance standards

---

## 🎯 **NEXT STEPS**

### **Immediate Actions**
1. **Complete Documentation**: Fill in all required sections
2. **Validate Information**: Ensure all information is accurate and current
3. **Test Procedures**: Verify all documented procedures work
4. **Update Regularly**: Keep documentation current with service changes

### **Ongoing Maintenance**
- **Monthly Review**: Review and update documentation monthly
- **Change Updates**: Update documentation with each service change
- **Feedback Integration**: Incorporate team feedback and suggestions
- **Compliance Monitoring**: Regular compliance checks and validation

---

**🎯 This template ensures comprehensive service documentation that meets all architecture requirements and provides the "five-second developer experience" for understanding any service.**

**💡 Pro Tip**: Use this template as a checklist to ensure your service documentation is complete and compliant with platform standards.
