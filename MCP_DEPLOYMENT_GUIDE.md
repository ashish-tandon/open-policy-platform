# 🚀 MCP Stack Complete Deployment Guide - Step by Step

## 📋 Prerequisites

Before starting, ensure you have:
- Docker & Docker Compose installed
- Python 3.9+ installed
- At least 8GB RAM available
- 20GB free disk space
- Basic command line knowledge

## 🎯 Step-by-Step Deployment Instructions

### Step 1: Clone and Setup Repository

```bash
# If you haven't cloned the repository
git clone https://github.com/your-org/OpenPolicyPlatformV5.git
cd OpenPolicyPlatformV5

# If you already have the repository
cd OpenPolicyPlatformV5
git pull origin main
```

### Step 2: Run Setup Script

```bash
# Make scripts executable (already done)
chmod +x scripts/setup-mcp-complete-40by6.sh
chmod +x scripts/deploy-complete-mcp-stack-40by6.sh

# Run the setup script
./scripts/setup-mcp-complete-40by6.sh
```

This script will:
- ✅ Check Python version
- ✅ Install Python dependencies
- ✅ Setup environment variables
- ✅ Create directory structure
- ✅ Initialize database
- ✅ Build Docker images
- ✅ Setup monitoring

**Expected output:**
```
[SETUP] Starting MCP Stack Complete Setup - 40by6 Implementation
[✓] Python 3.11 found
[✓] Python dependencies installed
[✓] Environment variables configured
[✓] Directory structure created
[✓] Docker network already exists
[✓] Database initialized
[✓] Docker images built
[✓] Monitoring configuration created
[✓] Initial tests passed
[✓] MCP Stack setup completed successfully!
```

### Step 3: Deploy MCP Stack Locally

```bash
# Deploy all services
./scripts/deploy-complete-mcp-stack-40by6.sh local deploy
```

This will:
- Start PostgreSQL and Redis
- Run database migrations
- Start MCP API service
- Start Scraper Orchestrator
- Start Scraper Workers
- Start Data Ingestion Pipeline
- Start Monitoring Stack
- Initialize Scraper Registry

**Expected output:**
```
[MCP] Deploying MCP Stack to local environment...
[MCP] Starting infrastructure services...
[MCP] Running database migrations...
[MCP] Starting all MCP services...
[MCP] Initializing scraper registry...
Initialized with 1732 scrapers
[✓] Local deployment complete!
```

### Step 4: Verify Deployment

```bash
# Run health check
./scripts/deploy-complete-mcp-stack-40by6.sh local health
```

**Expected output:**
```
[MCP] Running health checks...
[✓] MCP API is healthy
[✓] Scraper management is healthy
[✓] Monitoring is healthy
[MCP] System Status:
{
  "success_rate_24h": 0.0,
  "average_runtime": 0.0,
  "failed_scrapers": 0,
  "stale_scrapers": 0
}
```

### Step 5: Access Services

Open your browser and access:

1. **MCP API Documentation**
   ```
   http://localhost:8001/docs
   ```
   - Interactive API documentation
   - Test endpoints directly

2. **Scraper Dashboard**
   ```
   http://localhost:8001/api/v1/scrapers
   ```
   - View all 1700+ scrapers
   - Check scraper status
   - Execute scrapers

3. **Grafana Monitoring**
   ```
   http://localhost:3001
   ```
   - Username: `admin`
   - Password: `admin`
   - View real-time metrics

4. **Prometheus Metrics**
   ```
   http://localhost:9091
   ```
   - Raw metrics data
   - Query interface

### Step 6: Test Scraper Execution

```bash
# List all scrapers
curl http://localhost:8001/api/v1/scrapers | jq '.total'
# Output: 1732

# Get scraper statistics
curl http://localhost:8001/api/v1/scrapers/stats | jq '.'

# Execute federal parliament scrapers
curl -X POST http://localhost:8001/api/v1/scrapers/execute \
  -H "Content-Type: application/json" \
  -d '{"category": "federal_parliament"}'

# Check active runs
curl http://localhost:8001/api/v1/scrapers/runs/active | jq '.'
```

### Step 7: Monitor Execution

```bash
# View logs in real-time
./scripts/deploy-complete-mcp-stack-40by6.sh local logs

# View specific service logs
./scripts/deploy-complete-mcp-stack-40by6.sh local logs mcp-api
./scripts/deploy-complete-mcp-stack-40by6.sh local logs scraper-workers
```

### Step 8: Run Tests

```bash
# Run all tests
./scripts/deploy-complete-mcp-stack-40by6.sh local test

# Or run specific test suites
docker-compose -f docker-compose-mcp-40by6.yml run --rm mcp-api pytest tests/mcp/ -v
```

## 🎯 Common Operations

### Execute Scrapers by Category

```bash
# Federal Parliament
curl -X POST http://localhost:8001/api/v1/scrapers/execute \
  -H "Content-Type: application/json" \
  -d '{"category": "federal_parliament"}'

# Provincial Legislature (all provinces)
curl -X POST http://localhost:8001/api/v1/scrapers/execute \
  -H "Content-Type: application/json" \
  -d '{"category": "provincial_legislature"}'

# Municipal Councils
curl -X POST http://localhost:8001/api/v1/scrapers/execute \
  -H "Content-Type: application/json" \
  -d '{"category": "municipal_council"}'
```

### Check Scraper Health

```bash
# Run comprehensive health check
curl -X POST http://localhost:8001/api/v1/scrapers/health-check \
  -H "Content-Type: application/json" \
  -d '{
    "check_connectivity": true,
    "check_authentication": true,
    "check_data_quality": true,
    "sample_size": 10
  }'
```

### Export Scraper Registry

```bash
# Export complete scraper registry
curl http://localhost:8001/api/v1/scrapers/registry > scrapers_registry.json
```

### View Monitoring Dashboard

```bash
# Get dashboard data
curl http://localhost:8001/api/v1/scrapers/monitoring/dashboard | jq '.'
```

## 🛠️ Troubleshooting

### Services Not Starting

```bash
# Check Docker services
docker ps

# Check specific service logs
docker-compose -f docker-compose-mcp-40by6.yml logs postgres-mcp
docker-compose -f docker-compose-mcp-40by6.yml logs mcp-api

# Restart services
./scripts/deploy-complete-mcp-stack-40by6.sh local stop
./scripts/deploy-complete-mcp-stack-40by6.sh local deploy
```

### Database Connection Issues

```bash
# Check database is running
docker ps | grep postgres-mcp

# Test database connection
docker exec -it openpolicy-postgres-mcp-40by6 psql -U postgres -d openpolicy_mcp

# Recreate database
docker-compose -f docker-compose-mcp-40by6.yml down -v
./scripts/deploy-complete-mcp-stack-40by6.sh local deploy
```

### Scraper Execution Issues

```bash
# Check scraper worker logs
docker-compose -f docker-compose-mcp-40by6.yml logs scraper-workers

# Check Redis queue
docker exec -it openpolicy-redis-mcp-40by6 redis-cli
> LLEN scraper:queue
> LRANGE scraper:queue 0 10
```

## 🚀 Production Deployment

### Deploy to Kubernetes

```bash
# Deploy to staging
./scripts/deploy-complete-mcp-stack-40by6.sh staging deploy

# Deploy to production
./scripts/deploy-complete-mcp-stack-40by6.sh production deploy
```

### Scale Workers

```bash
# Scale scraper workers
kubectl scale deployment/scraper-workers --replicas=20 -n openpolicy-scrapers

# Enable autoscaling
kubectl autoscale deployment/scraper-workers \
  --min=5 --max=50 --cpu-percent=70 \
  -n openpolicy-scrapers
```

## 📊 Success Verification

Your MCP Stack is successfully deployed when:

1. ✅ All health checks pass
2. ✅ Scraper registry shows 1700+ scrapers
3. ✅ You can execute scrapers via API
4. ✅ Monitoring dashboards show metrics
5. ✅ Logs show successful scraper runs
6. ✅ Data is being ingested into database

## 🎉 Congratulations!

You have successfully deployed the complete MCP Stack with:
- 1700+ managed scrapers
- Real-time monitoring
- Automated scheduling
- Data quality assurance
- Full API access

The system is now ready for production use!

---

**Support**: If you encounter issues, check:
- Logs: `./scripts/deploy-complete-mcp-stack-40by6.sh local logs`
- Health: `./scripts/deploy-complete-mcp-stack-40by6.sh local health`
- Documentation: `/docs/mcp/`