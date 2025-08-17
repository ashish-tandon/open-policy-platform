# 🚀 Quick Start Guide - Open Policy Platform

## ⚡ **GET RUNNING IN 5 MINUTES**

> **Goal**: Start development with the Open Policy Platform in under 5 minutes

---

## 🎯 **PREREQUISITES CHECK**

### **Required Software**
- ✅ **Python 3.11+**: `python3 --version`
- ✅ **Docker**: `docker --version`
- ✅ **Docker Compose**: `docker compose version`
- ✅ **Git**: `git --version`

### **Quick Check Command**
```bash
# Run this to verify all prerequisites
python3 --version && docker --version && docker compose version && git --version
```

---

## 🚀 **STEP 1: CLONE & SETUP (1 minute)**

### **Clone Repository**
```bash
git clone https://github.com/ashish-tandon/open-policy-platform.git
cd open-policy-platform/open-policy-platform
```

### **Environment Setup**
```bash
# Copy environment template
cp env.example .env

# Edit environment file (use your preferred editor)
nano .env
# or
code .env
```

### **Essential Environment Variables**
```bash
# Database (required)
DATABASE_URL=postgresql://postgres:password@localhost:5432/openpolicy

# Security (required)
SECRET_KEY=your-secure-secret-key-change-this

# Environment
ENVIRONMENT=development
```

---

## 🚀 **STEP 2: START SERVICES (2 minutes)**

### **Start All Services**
```bash
# Start everything with one command
./scripts/start-all.sh
```

### **Alternative: Individual Services**
```bash
# Start database only
docker compose up -d postgres

# Start backend API
docker compose up -d api

# Start web interface
docker compose up -d web
```

### **Verify Services**
```bash
# Check service status
docker compose ps

# Check API health
curl http://localhost:8000/api/v1/health

# Check web interface
curl http://localhost:5173
```

---

## 🚀 **STEP 3: VERIFY INSTALLATION (1 minute)**

### **Access Points**
| Service | URL | Status Check |
|---------|-----|--------------|
| **Backend API** | http://localhost:8000 | `curl http://localhost:8000/api/v1/health` |
| **API Docs** | http://localhost:8000/docs | Open in browser |
| **Web Interface** | http://localhost:5173 | Open in browser |
| **Database** | localhost:5432 | `docker compose exec postgres psql -U postgres -d openpolicy` |

### **Quick Health Check**
```bash
# Comprehensive health check
curl -s http://localhost:8000/api/v1/health | jq .

# Expected response:
{
  "status": "healthy",
  "service": "Open Policy Platform API",
  "version": "1.0.0",
  "environment": "development",
  "timestamp": "2025-01-16T...",
  "uptime": "0:05:23"
}
```

---

## 🚀 **STEP 4: FIRST DEVELOPMENT TASK (1 minute)**

### **Make a Simple Change**
```bash
# Edit a file
nano backend/api/routers/health.py

# Add a comment or modify a response
# Save and restart the service
docker compose restart api
```

### **Test Your Change**
```bash
# Test the modified endpoint
curl http://localhost:8000/api/v1/health

# Check logs
docker compose logs api
```

---

## 🔧 **DEVELOPMENT WORKFLOW**

### **Daily Development Cycle**
```bash
# 1. Start services
./scripts/start-all.sh

# 2. Make changes
# Edit files in your preferred editor

# 3. Test changes
curl http://localhost:8000/api/v1/health

# 4. Restart if needed
docker compose restart api

# 5. Check logs
docker compose logs -f api
```

### **Quick Commands Reference**
```bash
# Service management
./scripts/start-all.sh          # Start everything
./scripts/stop-all.sh           # Stop everything
docker compose restart api       # Restart API only
docker compose logs -f api      # Follow API logs

# Development
./scripts/check-docs-links.sh   # Validate documentation
./scripts/export-openapi.sh     # Export API specification
./scripts/smoke-test.sh         # Run smoke tests
```

---

## 🐛 **TROUBLESHOOTING**

### **Common Issues & Solutions**

#### **1. Port Already in Use**
```bash
# Check what's using the port
lsof -i :8000
lsof -i :5173

# Kill the process or change ports in .env
```

#### **2. Database Connection Failed**
```bash
# Check database status
docker compose ps postgres

# Restart database
docker compose restart postgres

# Check logs
docker compose logs postgres
```

#### **3. Service Won't Start**
```bash
# Check all service status
docker compose ps

# Check service logs
docker compose logs api

# Restart everything
./scripts/stop-all.sh
./scripts/start-all.sh
```

#### **4. Environment Variables**
```bash
# Verify environment file
cat .env

# Check if variables are loaded
docker compose exec api env | grep DATABASE_URL
```

---

## 📚 **NEXT STEPS**

### **Immediate Actions**
1. **Explore API**: Visit http://localhost:8000/docs
2. **Check Web Interface**: Visit http://localhost:5173
3. **Review Code**: Browse the codebase structure
4. **Run Tests**: Execute test suite

### **Learning Path**
1. **API Development**: Study router implementations
2. **Database**: Explore models and migrations
3. **Frontend**: Understand React components
4. **Deployment**: Learn Docker and Kubernetes

### **Reference Materials**
- [**Common Commands**](./common-commands.md) - Daily development commands
- [**API Endpoints**](../api/endpoints.md) - Complete API reference
- [**Code Standards**](./code-standards.md) - Coding conventions
- [**Testing Guide**](./testing-guide.md) - Testing procedures

---

## 🎯 **SUCCESS CRITERIA**

### **✅ You're Ready When**
- [ ] All services are running (`docker compose ps` shows all healthy)
- [ ] API health check passes (`curl http://localhost:8000/api/v1/health`)
- [ ] Web interface loads (http://localhost:5173)
- [ ] API documentation accessible (http://localhost:8000/docs)
- [ ] Database connection working
- [ ] You can make and test a simple change

### **🚀 Ready to Develop**
- **Backend API**: FastAPI with comprehensive endpoints
- **Frontend**: React with TypeScript
- **Database**: PostgreSQL with 6.5GB parliamentary data
- **Monitoring**: Prometheus and Grafana
- **Containerization**: Docker with Kubernetes support

---

## 💡 **PRO TIPS**

### **Development Efficiency**
- **Use `./scripts/start-all.sh`** for quick service startup
- **Follow logs** with `docker compose logs -f service_name`
- **Hot reload** is enabled for development
- **Environment variables** in `.env` file for configuration

### **Quick Debugging**
- **Health endpoints** for service status
- **Logs** for detailed error information
- **API docs** for endpoint testing
- **Docker commands** for container management

---

## 🔗 **RELATED REFERENCE CARDS**

- [**Environment Setup**](./environment-setup.md) - Detailed environment configuration
- [**Common Commands**](./common-commands.md) - Daily development commands
- [**API Endpoints**](../api/endpoints.md) - Complete API reference
- [**Docker Commands**](../deployment/docker-commands.md) - Container management
- [**Health Checks**](../deployment/health-checks.md) - System monitoring

---

**🎉 Congratulations! You're now running the Open Policy Platform in development mode.**

**⏱️ Time to complete: Under 5 minutes**  
**🚀 Ready for development: Yes**  
**📚 Next step: Explore the API and start coding!**
