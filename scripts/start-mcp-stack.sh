#!/bin/bash

# 🚀 MCP Stack - One-Command Startup Script
# This script gets your complete MCP stack running in minutes!

set -e

echo "🚀 Starting MCP Stack - Complete AI-Powered Platform"
echo "=================================================="

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

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if ports are available
check_port() {
    local port=$1
    if lsof -i :$port > /dev/null 2>&1; then
        print_warning "Port $port is already in use. This might cause conflicts."
    else
        print_success "Port $port is available"
    fi
}

check_port 8000  # API
check_port 5173  # Web
check_port 5432  # Database
check_port 6379  # Redis

print_status "Prerequisites check complete!"

# Step 1: Setup MCP Stack
print_status "Step 1/3: Setting up MCP Stack..."
if [ -f "./scripts/setup-mcp-complete-40by6.sh" ]; then
    chmod +x ./scripts/setup-mcp-complete-40by6.sh
    ./scripts/setup-mcp-complete-40by6.sh
    print_success "MCP Stack setup complete!"
else
    print_error "Setup script not found. Please ensure you're in the correct directory."
    exit 1
fi

# Step 2: Deploy MCP Stack
print_status "Step 2/3: Deploying MCP Stack..."
if [ -f "./scripts/deploy-complete-mcp-stack-40by6.sh" ]; then
    chmod +x ./scripts/deploy-complete-mcp-stack-40by6.sh
    ./scripts/deploy-complete-mcp-stack-40by6.sh
    print_success "MCP Stack deployment complete!"
else
    print_error "Deployment script not found. Please ensure you're in the correct directory."
    exit 1
fi

# Step 3: Test Deployment
print_status "Step 3/3: Testing deployment..."
if [ -f "./scripts/test-mcp-deployment-40by6.sh" ]; then
    chmod +x ./scripts/test-mcp-deployment-40by6.sh
    ./scripts/test-mcp-deployment-40by6.sh
    print_success "Deployment testing complete!"
else
    print_error "Test script not found. Please ensure you're in the correct directory."
    exit 1
fi

# Final status check
print_status "Performing final validation..."
if [ -f "./scripts/validate-mcp-complete-40by6.sh" ]; then
    chmod +x ./scripts/validate-mcp-complete-40by6.sh
    ./scripts/validate-mcp-complete-40by6.sh
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
echo "💡 Need help? Check MCP_GETTING_STARTED_CHECKLIST.md"
echo "🚨 To stop everything: docker compose -f backend/docker-compose.yml down"
echo ""
print_success "Setup complete! Your MCP stack is ready to use!"
