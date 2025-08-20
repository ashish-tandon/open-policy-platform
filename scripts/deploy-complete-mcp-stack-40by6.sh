#!/bin/bash
# Complete MCP Stack Deployment Script - 40by6 Implementation
# Deploys all MCP components including scraper management system

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT=${1:-local}
ACTION=${2:-deploy}

# Function to print colored output
print_status() { echo -e "${BLUE}[MCP]${NC} $1"; }
print_success() { echo -e "${GREEN}[✓]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    local missing=()
    
    for cmd in docker docker-compose kubectl python3 npm redis-cli psql; do
        if ! command -v $cmd &> /dev/null; then
            missing+=($cmd)
        fi
    done
    
    if [ ${#missing[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing[*]}"
        return 1
    fi
    
    print_success "All prerequisites met"
    return 0
}

# Deploy to local environment
deploy_local() {
    print_status "Deploying MCP Stack to local environment..."
    
    # Start infrastructure
    print_status "Starting infrastructure services..."
    docker-compose -f docker-compose-mcp-40by6.yml up -d postgres-mcp redis-mcp
    
    # Wait for services
    print_status "Waiting for services to be ready..."
    sleep 10
    
    # Run database migrations
    print_status "Running database migrations..."
    docker-compose -f docker-compose-mcp-40by6.yml run --rm mcp-api python -m alembic upgrade head
    
    # Start all services
    print_status "Starting all MCP services..."
    docker-compose -f docker-compose-mcp-40by6.yml up -d
    
    # Wait for services to be healthy
    print_status "Waiting for services to be healthy..."
    sleep 20
    
    # Initialize scraper registry
    print_status "Initializing scraper registry..."
    docker-compose -f docker-compose-mcp-40by6.yml exec mcp-api python -c "
from backend.mcp.scraper_management_system import MCPScraperManagementSystem
import asyncio

async def init():
    system = MCPScraperManagementSystem()
    await system.initialize()
    print(f'Initialized with {len(system.registry.scrapers)} scrapers')
    system.registry.export_registry()

asyncio.run(init())
"
    
    print_success "Local deployment complete!"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    local env=$1
    print_status "Deploying MCP Stack to Kubernetes ($env)..."
    
    # Create namespace
    kubectl apply -f k8s/mcp/namespace-40by6.yaml
    kubectl apply -f k8s/mcp/scraper-workers-40by6.yaml
    
    # Apply configurations
    kubectl apply -k k8s/mcp/overlays/$env/
    
    # Wait for deployments
    print_status "Waiting for deployments to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment --all -n openpolicy-mcp
    kubectl wait --for=condition=available --timeout=300s deployment --all -n openpolicy-scrapers
    
    print_success "Kubernetes deployment complete!"
}

# Health check
health_check() {
    print_status "Running health checks..."
    
    # Check API health
    if curl -f http://localhost:8001/api/v1/health &>/dev/null; then
        print_success "MCP API is healthy"
    else
        print_error "MCP API health check failed"
        return 1
    fi
    
    # Check scraper management
    if curl -f http://localhost:8001/api/v1/scrapers/stats &>/dev/null; then
        print_success "Scraper management is healthy"
    else
        print_error "Scraper management health check failed"
        return 1
    fi
    
    # Check monitoring
    if curl -f http://localhost:3001/api/health &>/dev/null; then
        print_success "Monitoring is healthy"
    else
        print_warning "Monitoring health check failed (may still be starting)"
    fi
    
    # Show system status
    print_status "System Status:"
    curl -s http://localhost:8001/api/v1/scrapers/stats | jq '.health_metrics'
    
    return 0
}

# Run tests
run_tests() {
    print_status "Running MCP Stack tests..."
    
    # Run unit tests
    print_status "Running unit tests..."
    docker-compose -f docker-compose-mcp-40by6.yml run --rm mcp-api pytest tests/mcp/unit/ -v
    
    # Run integration tests
    print_status "Running integration tests..."
    docker-compose -f docker-compose-mcp-40by6.yml run --rm mcp-api pytest tests/mcp/integration/ -v
    
    print_success "All tests passed!"
}

# Show logs
show_logs() {
    local service=${1:-}
    
    if [ -z "$service" ]; then
        docker-compose -f docker-compose-mcp-40by6.yml logs -f
    else
        docker-compose -f docker-compose-mcp-40by6.yml logs -f $service
    fi
}

# Stop services
stop_services() {
    print_status "Stopping MCP Stack services..."
    
    if [ "$ENVIRONMENT" == "local" ]; then
        docker-compose -f docker-compose-mcp-40by6.yml down
    else
        kubectl delete -k k8s/mcp/overlays/$ENVIRONMENT/
    fi
    
    print_success "Services stopped"
}

# Main deployment function
main() {
    print_status "MCP Stack Deployment - 40by6 Implementation"
    print_status "Environment: $ENVIRONMENT"
    print_status "Action: $ACTION"
    
    # Check prerequisites
    if ! check_prerequisites; then
        print_error "Prerequisites check failed"
        exit 1
    fi
    
    case $ACTION in
        deploy)
            case $ENVIRONMENT in
                local)
                    deploy_local
                    health_check
                    ;;
                staging|production)
                    deploy_kubernetes $ENVIRONMENT
                    ;;
                *)
                    print_error "Unknown environment: $ENVIRONMENT"
                    exit 1
                    ;;
            esac
            ;;
        
        test)
            run_tests
            ;;
        
        logs)
            show_logs $3
            ;;
        
        stop)
            stop_services
            ;;
        
        health)
            health_check
            ;;
        
        *)
            print_error "Unknown action: $ACTION"
            echo "Usage: $0 [local|staging|production] [deploy|test|logs|stop|health]"
            exit 1
            ;;
    esac
    
    if [ "$ACTION" == "deploy" ]; then
        print_success "MCP Stack deployment completed successfully!"
        print_status "Access points:"
        echo "  - MCP API: http://localhost:8001"
        echo "  - Scraper Dashboard: http://localhost:8001/api/v1/scrapers"
        echo "  - Grafana: http://localhost:3001 (admin/admin)"
        echo "  - Prometheus: http://localhost:9091"
        echo ""
        print_status "Quick commands:"
        echo "  - View logs: $0 $ENVIRONMENT logs"
        echo "  - Run tests: $0 $ENVIRONMENT test"
        echo "  - Health check: $0 $ENVIRONMENT health"
        echo "  - Stop services: $0 $ENVIRONMENT stop"
        echo ""
        print_status "Scraper management:"
        echo "  - List scrapers: curl http://localhost:8001/api/v1/scrapers"
        echo "  - View stats: curl http://localhost:8001/api/v1/scrapers/stats"
        echo "  - Execute scrapers: curl -X POST http://localhost:8001/api/v1/scrapers/execute"
    fi
}

# Run main function
main "$@"