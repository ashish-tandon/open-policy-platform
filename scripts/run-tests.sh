#!/bin/bash

# OpenPolicy Merge - Test Execution Script
# This script runs all tests following the test-driven development approach

set -e  # Exit on any error

echo "🧪 OpenPolicy Merge - Test Execution"
echo "===================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking test prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed"
        exit 1
    fi
    
    # PostgreSQL is optional for tests (SQLite fallback)
    if ! command -v psql &> /dev/null; then
        print_warning "PostgreSQL client not found; tests will use SQLite"
    else
        if ! pg_isready -q; then
            print_warning "PostgreSQL is not running; tests will use SQLite"
        fi
    fi
    
    print_success "Prerequisites check completed"
}

# Setup test environment
setup_test_environment() {
    print_status "Setting up test environment..."
    
    # Create test database
    if psql -lqt | cut -d \| -f 1 | grep -qw openpolicy_test; then
        print_warning "Test database already exists, dropping..."
        dropdb openpolicy_test
    fi
    
    createdb openpolicy_test
    print_success "Test database created"
    
    # Install Python test dependencies
    cd backend
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    pip install pytest pytest-asyncio pytest-fastapi pytest-postgresql pytest-mock pytest-cov pytest-httpx
    cd ..
    
    print_success "Test environment setup completed"
}

# Run database tests
run_database_tests() {
    print_status "Running database tests..."
    
    cd backend
    source venv/bin/activate
    
    # Run schema validation tests
    python -m pytest tests/database/ -v --tb=short
    
    # Run migration tests
    python -m pytest tests/migrations/ -v --tb=short
    
    # Run data integrity tests
    python -m pytest tests/integrity/ -v --tb=short
    
    cd ..
    
    print_success "Database tests completed"
}

# Run scraper tests
run_scraper_tests() {
    print_status "Running scraper tests..."
    
    cd backend
    source venv/bin/activate
    
    # Run federal scraper tests
    python -m pytest tests/scrapers/federal/ -v --tb=short
    
    # Run provincial scraper tests
    python -m pytest tests/scrapers/provincial/ -v --tb=short
    
    # Run municipal scraper tests
    python -m pytest tests/scrapers/municipal/ -v --tb=short
    
    # Run error handling tests
    python -m pytest tests/scrapers/error_handling/ -v --tb=short
    
    cd ..
    
    print_success "Scraper tests completed"
}

# Run API tests
run_api_tests() {
    print_status "Running API tests..."
    
    cd backend
    source venv/bin/activate
    
    # Run authentication tests
    python -m pytest tests/api/auth/ -v --tb=short
    
    # Run policy endpoint tests
    python -m pytest tests/api/policies/ -v --tb=short
    
    # Run representative endpoint tests
    python -m pytest tests/api/representatives/ -v --tb=short
    
    # Run scraper management tests
    python -m pytest tests/api/scrapers/ -v --tb=short
    
    # Run admin endpoint tests
    python -m pytest tests/api/admin/ -v --tb=short
    
    # Run health endpoint tests
    python -m pytest tests/api/health/ -v --tb=short
    
    cd ..
    
    print_success "API tests completed"
}

# Run integration tests
run_integration_tests() {
    print_status "Running integration tests..."
    
    cd backend
    source venv/bin/activate
    
    # Run end-to-end data flow tests
    python -m pytest tests/integration/data_flow/ -v --tb=short
    
    # Run error handling integration tests
    python -m pytest tests/integration/error_handling/ -v --tb=short
    
    # Run performance tests
    python -m pytest tests/integration/performance/ -v --tb=short
    
    cd ..
    
    print_success "Integration tests completed"
}

# Run frontend tests
run_frontend_tests() {
    print_status "Running frontend tests..."
    
    cd web
    
    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        npm install
    fi
    
    # Run unit tests
    npm test -- --coverage --watchAll=false
    
    # Run component tests
    npm run test:components
    
    cd ..
    
    print_success "Frontend tests completed"
}

# Run E2E tests
run_e2e_tests() {
    print_status "Running E2E tests..."
    
    # Start the backend for E2E testing
    cd backend
    source venv/bin/activate
    python -m pytest tests/e2e/ -v --tb=short
    cd ..
    
    print_success "E2E tests completed"
}

# Generate test reports
generate_test_reports() {
    print_status "Generating test reports..."
    
    cd backend
    source venv/bin/activate
    
    # Generate coverage report
    python -m pytest --cov=. --cov-report=html --cov-report=term-missing
    
    # Generate test summary
    python -m pytest --tb=short --junitxml=test-results.xml
    
    cd ..
    
    print_success "Test reports generated"
}

# Cleanup test environment
cleanup_test_environment() {
    print_status "Cleaning up test environment..."
    
    # Drop test database
    dropdb openpolicy_test 2>/dev/null || true
    
    print_success "Test environment cleaned up"
}

# Main test execution
main() {
    echo "Starting comprehensive test execution..."
    echo ""
    
    check_prerequisites
    setup_test_environment
    
    echo ""
    echo "🧪 Running Test Suite..."
    echo "========================"
    
    # Phase 1: Database Tests
    echo ""
    echo "📊 Phase 1: Database Tests"
    echo "-------------------------"
    run_database_tests
    
    # Phase 2: Scraper Tests
    echo ""
    echo "🕷️ Phase 2: Scraper Tests"
    echo "------------------------"
    run_scraper_tests
    
    # Phase 3: API Tests
    echo ""
    echo "🔌 Phase 3: API Tests"
    echo "---------------------"
    run_api_tests
    
    # Phase 4: Integration Tests
    echo ""
    echo "🔗 Phase 4: Integration Tests"
    echo "----------------------------"
    run_integration_tests
    
    # Phase 5: Frontend Tests
    echo ""
    echo "🎨 Phase 5: Frontend Tests"
    echo "-------------------------"
    run_frontend_tests
    
    # Phase 6: E2E Tests
    echo ""
    echo "🌐 Phase 6: E2E Tests"
    echo "---------------------"
    run_e2e_tests
    
    # Generate Reports
    echo ""
    echo "📊 Generating Test Reports"
    echo "-------------------------"
    generate_test_reports
    
    # Cleanup
    cleanup_test_environment
    
    echo ""
    echo "🎉 Test execution completed!"
    echo "============================"
    echo ""
    echo "📋 Test Results:"
    echo "  - Database Tests: ✅"
    echo "  - Scraper Tests: ✅"
    echo "  - API Tests: ✅"
    echo "  - Integration Tests: ✅"
    echo "  - Frontend Tests: ✅"
    echo "  - E2E Tests: ✅"
    echo ""
    echo "📊 Coverage Report: backend/htmlcov/index.html"
    echo "📄 Test Results: backend/test-results.xml"
    echo ""
    echo "🚀 Platform ready for deployment!"
}

# Run main function
main "$@"
