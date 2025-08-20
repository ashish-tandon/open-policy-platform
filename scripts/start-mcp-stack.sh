#!/bin/bash

# 🚀 MCP Stack - One-Command Startup Script
# This script gets your complete MCP stack running in minutes!
# 
# 📁 File Locations:
# - Main script: scripts/start-mcp-stack.sh
# - Setup script: scripts/setup-mcp-complete-40by6.sh
# - Deployment script: scripts/deploy-complete-mcp-stack-40by6.sh
# - Test script: scripts/test-mcp-deployment-40by6.sh
# - Validation script: scripts/validate-mcp-complete-40by6.sh
# - Docker compose: backend/docker-compose.yml
# - MCP modules: backend/mcp/ (40+ modules)
# - Web components: web/src/components/
# - Mobile app: mobile/App.tsx

set -e

echo "🚀 Starting MCP Stack - Complete AI-Powered Platform"
echo "=================================================="
echo ""
echo "📁 This script will deploy from the following locations:"
echo "   • Scripts: scripts/"
echo "   • Backend: backend/"
echo "   • MCP Modules: backend/mcp/"
echo "   • Web App: web/"
echo "   • Mobile App: mobile/"
echo "   • Kubernetes: k8s/mcp/"
echo "   • Documentation: docs/ and root directory"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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
print_status "Checking prerequisites..."

# Check if we're in the right directory
if [ ! -f "scripts/start-mcp-stack.sh" ]; then
    print_error "This script must be run from the root directory of the open-policy-platform repository."
    print_error "Current directory: $(pwd)"
    print_error "Expected files: scripts/start-mcp-stack.sh, backend/, web/, mobile/"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    print_error "Download Docker from: https://docker.com"
    exit 1
fi

# Check if required directories exist
required_dirs=("backend" "web" "mobile" "scripts" "k8s")
for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        print_error "Required directory '$dir' not found. Please ensure you're in the correct repository."
        print_error "Current directory contents: $(ls -la)"
        exit 1
    fi
done

# Check if required files exist
required_files=("scripts/setup-mcp-complete-40by6.sh" "scripts/deploy-complete-mcp-stack-40by6.sh" "backend/docker-compose.yml")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        print_error "Required file '$file' not found. Please ensure you have the complete repository."
        exit 1
    fi
done

# Check if ports are available
check_port() {
    local port=$1
    local service_name=$2
    if lsof -i :$port > /dev/null 2>&1; then
        print_warning "Port $port ($service_name) is already in use. This might cause conflicts."
        print_warning "Current processes using port $port:"
        lsof -i :$port
        echo ""
    else
        print_success "Port $port ($service_name) is available"
    fi
}

print_status "Checking port availability..."
check_port 8000 "API Backend (FastAPI)"
check_port 5173 "Web Frontend (React)"
check_port 5432 "PostgreSQL Database"
check_port 6379 "Redis Cache"

# Check available memory
print_status "Checking system resources..."
if command -v free > /dev/null 2>&1; then
    available_mem=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    if [ "$available_mem" -lt 4000 ]; then
        print_warning "Available memory: ${available_mem}MB (recommended: 4GB+ for smooth operation)"
    else
        print_success "Available memory: ${available_mem}MB (sufficient for MCP stack)"
    fi
else
    print_warning "Cannot check memory usage. Ensure you have at least 4GB available."
fi

print_status "Prerequisites check complete!"

# Step 1: Setup MCP Stack
print_status "Step 1/3: Setting up MCP Stack..."
print_status "Running: ./scripts/setup-mcp-complete-40by6.sh"
if [ -f "./scripts/setup-mcp-complete-40by6.sh" ]; then
    chmod +x ./scripts/setup-mcp-complete-40by6.sh
    ./scripts/setup-mcp-complete-40by6.sh
    print_success "MCP Stack setup complete!"
else
    print_error "Setup script not found at: ./scripts/setup-mcp-complete-40by6.sh"
    print_error "Please ensure you're in the correct directory and have the complete repository."
    exit 1
fi

# Step 2: Deploy MCP Stack
print_status "Step 2/3: Deploying MCP Stack..."
print_status "Running: ./scripts/deploy-complete-mcp-stack-40by6.sh"
if [ -f "./scripts/deploy-complete-mcp-stack-40by6.sh" ]; then
    chmod +x ./scripts/deploy-complete-mcp-stack-40by6.sh
    ./scripts/deploy-complete-mcp-stack-40by6.sh
    print_success "MCP Stack deployment complete!"
else
    print_error "Deployment script not found at: ./scripts/deploy-complete-mcp-stack-40by6.sh"
    print_error "Please ensure you're in the correct directory and have the complete repository."
    exit 1
fi

# Step 3: Test Deployment
print_status "Step 3/3: Testing deployment..."
print_status "Running: ./scripts/test-mcp-deployment-40by6.sh"
if [ -f "./scripts/test-mcp-deployment-40by6.sh" ]; then
    chmod +x ./scripts/test-mcp-deployment-40by6.sh
    ./scripts/test-mcp-deployment-40by6.sh
    print_success "Deployment testing complete!"
else
    print_error "Test script not found at: ./scripts/test-mcp-deployment-40by6.sh"
    print_error "Please ensure you're in the correct directory and have the complete repository."
    exit 1
fi

# Final status check
print_status "Performing final validation..."
print_status "Running: ./scripts/validate-mcp-complete-40by6.sh"
if [ -f "./scripts/validate-mcp-complete-40by6.sh" ]; then
    chmod +x ./scripts/validate-mcp-complete-40by6.sh
    ./scripts/validate-mcp-complete-40by6.sh
else
    print_warning "Validation script not found. Proceeding with basic checks..."
    # Basic health checks
    print_status "Performing basic health checks..."
    
    # Check if services are running
    if command -v docker > /dev/null 2>&1; then
        print_status "Checking Docker services status..."
        docker compose -f backend/docker-compose.yml ps
        
        print_status "Checking service logs for errors..."
        docker compose -f backend/docker-compose.yml logs --tail=10
    fi
fi

echo ""
echo "🎉 MCP Stack is now running!"
echo "============================"
echo ""
echo "🌐 Access your platform at:"
echo "   • Web Dashboard: http://localhost:5173"
echo "   • API Backend:  http://localhost:8000"
echo "   • Health Check: http://localhost:8000/api/v1/health"
echo ""
echo "📱 Mobile app components are ready in mobile/App.tsx"
echo "🔧 Admin panel available at: http://localhost:5173/admin"
echo "📊 Scraper dashboard at: http://localhost:5173/scrapers"
echo ""
echo "📁 Key file locations:"
echo "   • MCP Modules: backend/mcp/ (40+ AI modules)"
echo "   • Web Components: web/src/components/"
echo "   • API Backend: backend/api/"
echo "   • Docker Config: backend/docker-compose.yml"
echo "   • Kubernetes: k8s/mcp/"
echo "   • Documentation: docs/ and root directory"
echo ""
echo "💡 Need help? Check MCP_GETTING_STARTED_CHECKLIST.md"
echo "🚨 To stop everything: docker compose -f backend/docker-compose.yml down"
echo "🔄 To restart: docker compose -f backend/docker-compose.yml restart"
echo "📋 To check status: docker compose -f backend/docker-compose.yml ps"
echo "📝 To view logs: docker compose -f backend/docker-compose.yml logs -f"
echo ""
print_success "Setup complete! Your MCP stack is ready to use!"
echo ""
echo "🎯 Next steps:"
echo "   1. Visit http://localhost:5173 to explore the web dashboard"
echo "   2. Check http://localhost:8000/api/v1/health for API status"
echo "   3. Explore MCP modules in backend/mcp/ directory"
echo "   4. Customize components in web/src/components/"
echo "   5. Build mobile app from mobile/ directory"
