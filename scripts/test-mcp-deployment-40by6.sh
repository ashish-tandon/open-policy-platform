#!/bin/bash
# MCP Stack Deployment Test Script - 40by6
# Validates that all components are working correctly

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
API_URL="http://localhost:8001"
GRAFANA_URL="http://localhost:3001"
PROMETHEUS_URL="http://localhost:9091"

# Test results
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test
run_test() {
    local test_name=$1
    local test_command=$2
    
    echo -n "Testing $test_name... "
    
    if eval "$test_command" &>/dev/null; then
        echo -e "${GREEN}PASSED${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}FAILED${NC}"
        ((TESTS_FAILED++))
    fi
}

# Function to check endpoint
check_endpoint() {
    local name=$1
    local url=$2
    local expected_code=${3:-200}
    
    run_test "$name" "curl -s -o /dev/null -w '%{http_code}' '$url' | grep -q '$expected_code'"
}

echo "🧪 MCP Stack Deployment Tests - 40by6"
echo "====================================="

# 1. Infrastructure Tests
echo -e "\n📦 Infrastructure Tests:"
run_test "PostgreSQL MCP" "docker ps | grep -q postgres-mcp"
run_test "Redis MCP" "docker ps | grep -q redis-mcp"
run_test "MCP API Container" "docker ps | grep -q mcp-api"
run_test "Docker Network" "docker network ls | grep -q openpolicy-mcp-network"

# 2. API Endpoint Tests
echo -e "\n🌐 API Endpoint Tests:"
check_endpoint "Health Check" "$API_URL/api/v1/health"
check_endpoint "MCP Health" "$API_URL/api/v1/mcp/health"
check_endpoint "Scraper Stats" "$API_URL/api/v1/scrapers/stats"
check_endpoint "API Documentation" "$API_URL/docs"
check_endpoint "OpenAPI Schema" "$API_URL/openapi.json"

# 3. Scraper Management Tests
echo -e "\n📊 Scraper Management Tests:"
run_test "Scraper Count" "curl -s '$API_URL/api/v1/scrapers' | jq '.total' | grep -E '^[0-9]+$'"
run_test "Scraper Categories" "curl -s '$API_URL/api/v1/scrapers/stats' | jq '.by_category' | grep -q 'federal_parliament'"
run_test "Scraper Platforms" "curl -s '$API_URL/api/v1/scrapers/stats' | jq '.by_platform' | grep -q 'legistar'"
run_test "Jurisdictions" "curl -s '$API_URL/api/v1/scrapers/jurisdictions' | jq '.total_jurisdictions' | grep -E '^[0-9]+$'"

# 4. Data Quality Tests
echo -e "\n✅ Data Quality Tests:"
run_test "Quality Report" "curl -s '$API_URL/api/v1/mcp/quality' | jq '.timestamp' | grep -q '20'"
run_test "Agent Status" "curl -s '$API_URL/api/v1/mcp/agent/status' | jq '.agent_status' | grep -q 'running'"

# 5. Monitoring Tests
echo -e "\n📈 Monitoring Tests:"
check_endpoint "Prometheus" "$PROMETHEUS_URL"
check_endpoint "Grafana" "$GRAFANA_URL/api/health"
run_test "Metrics Endpoint" "curl -s '$API_URL/metrics' | grep -q 'http_requests_total'"

# 6. Database Tests
echo -e "\n🗄️ Database Tests:"
run_test "Database Connection" "docker exec openpolicy-postgres-mcp-40by6 psql -U postgres -d openpolicy_mcp -c 'SELECT 1' | grep -q '1 row'"
run_test "Tables Created" "docker exec openpolicy-postgres-mcp-40by6 psql -U postgres -d openpolicy_mcp -c '\dt' | grep -q 'scrapers'"

# 7. Functional Tests
echo -e "\n⚡ Functional Tests:"
# Test scraper discovery
run_test "Scraper Discovery" "curl -s -X POST '$API_URL/api/v1/scrapers/discover' | jq '.status' | grep -q 'discovery_started'"

# Test scraper execution (dry run)
run_test "Scraper Execution API" "curl -s -X POST '$API_URL/api/v1/scrapers/execute' -H 'Content-Type: application/json' -d '{\"scraper_ids\": []}' | jq '.status' | grep -q 'scheduled'"

# Test active runs
run_test "Active Runs API" "curl -s '$API_URL/api/v1/scrapers/runs/active' | jq '.active_count' | grep -E '^[0-9]+$'"

# 8. Performance Tests
echo -e "\n🚀 Performance Tests:"
# Test API response time
RESPONSE_TIME=$(curl -s -o /dev/null -w '%{time_total}' "$API_URL/api/v1/health")
if (( $(echo "$RESPONSE_TIME < 0.1" | bc -l) )); then
    echo -e "API Response Time (<100ms)... ${GREEN}PASSED${NC} (${RESPONSE_TIME}s)"
    ((TESTS_PASSED++))
else
    echo -e "API Response Time (<100ms)... ${YELLOW}WARNING${NC} (${RESPONSE_TIME}s)"
fi

# 9. Integration Tests
echo -e "\n🔗 Integration Tests:"
# Test full workflow
run_test "Registry Export" "curl -s '$API_URL/api/v1/scrapers/registry' | jq '.total_scrapers' | grep -E '^[0-9]+$'"
run_test "Dashboard Data" "curl -s '$API_URL/api/v1/scrapers/monitoring/dashboard' | jq '.overview' | grep -q 'timestamp'"

# Summary
echo -e "\n====================================="
echo "📊 Test Summary:"
echo -e "   Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "   Failed: ${RED}$TESTS_FAILED${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "\n${GREEN}✅ All tests passed! MCP Stack is fully operational.${NC}"
    echo -e "\n🎉 Deployment Statistics:"
    curl -s "$API_URL/api/v1/scrapers/stats" | jq '{
        total_scrapers: .total_scrapers,
        categories: .by_category | length,
        platforms: .by_platform | length,
        active_scrapers: .by_status.active // 0
    }'
    exit 0
else
    echo -e "\n${RED}❌ Some tests failed. Please check the logs for details.${NC}"
    echo -e "\nDebug commands:"
    echo "  - Check logs: docker-compose -f docker-compose-mcp-40by6.yml logs"
    echo "  - Check containers: docker ps"
    echo "  - Check API: curl -v $API_URL/api/v1/health"
    exit 1
fi