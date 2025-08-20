#!/bin/bash
# MCP Stack Complete Setup Script - 40by6 Implementation
# Sets up the entire MCP infrastructure including scraper management

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
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Function to print colored output
print_status() { echo -e "${BLUE}[SETUP]${NC} $1"; }
print_success() { echo -e "${GREEN}[✓]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }

# Check Python version
check_python() {
    print_status "Checking Python version..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        if [[ $(echo "$PYTHON_VERSION >= 3.9" | bc) -eq 1 ]]; then
            print_success "Python $PYTHON_VERSION found"
        else
            print_error "Python 3.9+ required, found $PYTHON_VERSION"
            return 1
        fi
    else
        print_error "Python 3 not found"
        return 1
    fi
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$PROJECT_ROOT/venv" ]; then
        python3 -m venv "$PROJECT_ROOT/venv"
        print_success "Created virtual environment"
    fi
    
    # Activate virtual environment
    source "$PROJECT_ROOT/venv/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        pip install -r "$PROJECT_ROOT/requirements.txt"
    fi
    
    if [ -f "$PROJECT_ROOT/backend/requirements.txt" ]; then
        pip install -r "$PROJECT_ROOT/backend/requirements.txt"
    fi
    
    # Install additional MCP dependencies
    pip install croniter redis aiofiles aiohttp psutil
    
    print_success "Python dependencies installed"
}

# Setup environment variables
setup_environment() {
    print_status "Setting up environment variables..."
    
    # Create .env file if it doesn't exist
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        cp "$PROJECT_ROOT/env.example" "$PROJECT_ROOT/.env"
        print_success "Created .env file from template"
    fi
    
    # Add MCP-specific variables
    cat >> "$PROJECT_ROOT/.env" << EOF

# MCP Stack Configuration - 40by6
MCP_ENABLED=true
MCP_AGENT_INTERVAL=300
MCP_AUTO_REMEDIATE=false
MAX_CONCURRENT_SCRAPERS=20
SCRAPER_TIMEOUT=300
RETRY_COUNT=3
RATE_LIMIT_REQUESTS=10
RATE_LIMIT_PERIOD=60
SCHEDULE_TIMEZONE=America/Toronto
EOF
    
    print_success "Environment variables configured"
}

# Create necessary directories
create_directories() {
    print_status "Creating directory structure..."
    
    directories=(
        "logs/mcp"
        "reports/mcp"
        "backend/mcp/services"
        "tests/mcp/unit"
        "tests/mcp/integration"
        "tests/mcp/e2e"
        "k8s/mcp/overlays/dev"
        "k8s/mcp/overlays/staging"
        "k8s/mcp/overlays/production"
        "monitoring/mcp/grafana/dashboards"
        "monitoring/mcp/prometheus/rules"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$PROJECT_ROOT/$dir"
    done
    
    print_success "Directory structure created"
}

# Setup Docker networks
setup_docker_networks() {
    print_status "Setting up Docker networks..."
    
    # Create network if it doesn't exist
    if ! docker network ls | grep -q "openpolicy-mcp-network"; then
        docker network create openpolicy-mcp-network
        print_success "Created Docker network: openpolicy-mcp-network"
    else
        print_success "Docker network already exists"
    fi
}

# Initialize database
initialize_database() {
    print_status "Initializing database..."
    
    # Start PostgreSQL if not running
    if ! docker ps | grep -q "postgres"; then
        docker-compose -f "$PROJECT_ROOT/docker-compose.yml" up -d postgres
        sleep 10
    fi
    
    # Create MCP database
    docker exec -i $(docker ps -qf "name=postgres") psql -U postgres << EOF
CREATE DATABASE IF NOT EXISTS openpolicy_mcp;
GRANT ALL PRIVILEGES ON DATABASE openpolicy_mcp TO postgres;
EOF
    
    print_success "Database initialized"
}

# Build Docker images
build_docker_images() {
    print_status "Building Docker images..."
    
    # Build MCP images
    docker build -t openpolicy/mcp-api:latest -f backend/Dockerfile --target api .
    docker build -t openpolicy/mcp-agent:latest -f backend/Dockerfile --target agent .
    docker build -t openpolicy/scraper-worker:latest -f backend/Dockerfile --target worker .
    
    print_success "Docker images built"
}

# Generate API documentation
generate_api_docs() {
    print_status "Generating API documentation..."
    
    # Export OpenAPI schema
    if [ -x "$PROJECT_ROOT/scripts/export-openapi.sh" ]; then
        "$PROJECT_ROOT/scripts/export-openapi.sh"
        print_success "API documentation generated"
    else
        print_warning "export-openapi.sh not found or not executable"
    fi
}

# Setup monitoring
setup_monitoring() {
    print_status "Setting up monitoring configuration..."
    
    # Create Prometheus configuration
    cat > "$PROJECT_ROOT/monitoring/mcp/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'mcp-api'
    static_configs:
      - targets: ['mcp-api:8000']
    metrics_path: '/metrics'

  - job_name: 'scrapers'
    static_configs:
      - targets: ['scraper-orchestrator:8080']
EOF
    
    # Create Grafana datasource
    mkdir -p "$PROJECT_ROOT/monitoring/mcp/grafana/provisioning/datasources"
    cat > "$PROJECT_ROOT/monitoring/mcp/grafana/provisioning/datasources/prometheus.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus-MCP
    type: prometheus
    access: proxy
    url: http://prometheus-mcp:9090
    isDefault: true
EOF
    
    print_success "Monitoring configuration created"
}

# Run initial tests
run_initial_tests() {
    print_status "Running initial tests..."
    
    # Activate virtual environment
    source "$PROJECT_ROOT/venv/bin/activate"
    
    # Run basic import tests
    python3 -c "
import backend.mcp.data_quality_agent
import backend.mcp.scraper_management_system
import backend.mcp.scraper_scheduler
print('✓ All MCP modules imported successfully')
"
    
    print_success "Initial tests passed"
}

# Main setup function
main() {
    print_status "Starting MCP Stack Complete Setup - 40by6 Implementation"
    print_status "Project root: $PROJECT_ROOT"
    print_status "Timestamp: $TIMESTAMP"
    
    # Run setup steps
    check_python || exit 1
    install_python_deps
    setup_environment
    create_directories
    setup_docker_networks
    initialize_database
    build_docker_images
    generate_api_docs
    setup_monitoring
    run_initial_tests
    
    print_success "MCP Stack setup completed successfully!"
    print_status "Next steps:"
    echo "  1. Start services: ./scripts/deploy-complete-mcp-stack-40by6.sh local deploy"
    echo "  2. Run health check: ./scripts/deploy-complete-mcp-stack-40by6.sh local health"
    echo "  3. View logs: ./scripts/deploy-complete-mcp-stack-40by6.sh local logs"
    echo "  4. Access services:"
    echo "     - MCP API: http://localhost:8001"
    echo "     - Grafana: http://localhost:3001 (admin/admin)"
    echo "     - Prometheus: http://localhost:9091"
    echo ""
    print_status "For production deployment:"
    echo "  ./scripts/deploy-complete-mcp-stack-40by6.sh production deploy"
}

# Run main function
main "$@"