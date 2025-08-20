#!/bin/bash

# 🚀 Azure MCP Stack Deployment Script
# This script deploys your complete MCP stack to Azure

set -e

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

echo "🚀 Azure MCP Stack Deployment"
echo "============================="
echo ""

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    print_error "Azure CLI is not installed. Please install it first."
    exit 1
fi

# Check if logged in to Azure
if ! az account show &> /dev/null; then
    print_status "You need to log in to Azure first."
    print_status "Running: az login"
    az login
fi

# Configuration
RESOURCE_GROUP="mcp-stack-rg"
LOCATION="eastus"
ACR_NAME="mcpstackacr$(date +%s | tail -c 5)"
CONTAINER_APP_ENV="mcp-stack-env"
APP_NAME="mcp-stack-api"
WEB_APP_NAME="mcp-stack-web"

print_status "Setting up Azure resources for MCP Stack deployment..."

# Create Resource Group
print_status "Creating resource group: $RESOURCE_GROUP"
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Azure Container Registry
print_status "Creating Azure Container Registry: $ACR_NAME"
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --admin-enabled true

# Get ACR credentials
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP --query "loginServer" --output tsv)
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query "username" --output tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" --output tsv)

print_success "ACR created: $ACR_LOGIN_SERVER"

# Create Container Apps Environment
print_status "Creating Container Apps Environment: $CONTAINER_APP_ENV"
az containerapp env create \
    --name $CONTAINER_APP_ENV \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION

# Create PostgreSQL Flexible Server
print_status "Creating PostgreSQL Flexible Server"
POSTGRES_SERVER="mcp-postgres-$(date +%s | tail -c 5)"
POSTGRES_USER="mcpadmin"
POSTGRES_PASSWORD="MCP$(date +%s | tail -c 8)!"

az postgres flexible-server create \
    --resource-group $RESOURCE_GROUP \
    --name $POSTGRES_SERVER \
    --location $LOCATION \
    --admin-user $POSTGRES_USER \
    --admin-password $POSTGRES_PASSWORD \
    --sku-name "Standard_B1ms" \
    --tier "Burstable" \
    --storage-size 32 \
    --version "15"

# Create Redis Cache
print_status "Creating Redis Cache"
REDIS_NAME="mcp-redis-$(date +%s | tail -c 5)"

az redis create \
    --resource-group $RESOURCE_GROUP \
    --name $REDIS_NAME \
    --location $LOCATION \
    --sku Basic \
    --vm-size C0

# Get Redis connection details
REDIS_HOSTNAME=$(az redis show --name $REDIS_NAME --resource-group $RESOURCE_GROUP --query "hostName" --output tsv)
REDIS_SSL_PORT=$(az redis show --name $REDIS_NAME --resource-group $RESOURCE_GROUP --query "sslPort" --output tsv)
REDIS_ACCESS_KEY=$(az redis list-keys --name $REDIS_NAME --resource-group $RESOURCE_GROUP --query "primaryKey" --output tsv)

print_success "Redis Cache created: $REDIS_HOSTNAME"

# Build and push Docker images
print_status "Building and pushing Docker images to ACR..."

# Login to ACR
az acr login --name $ACR_NAME

# Build and push backend image
print_status "Building backend image..."
cd backend
az acr build --registry $ACR_NAME --image mcp-backend:latest .

# Build and push web image
print_status "Building web image..."
cd ../web
az acr build --registry $ACR_NAME --image mcp-web:latest .

cd ..

# Create Container App for Backend API
print_status "Creating Container App for Backend API: $APP_NAME"

az containerapp create \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --environment $CONTAINER_APP_ENV \
    --image $ACR_LOGIN_SERVER/mcp-backend:latest \
    --target-port 8000 \
    --ingress external \
    --registry-server $ACR_LOGIN_SERVER \
    --registry-username $ACR_USERNAME \
    --registry-password $ACR_PASSWORD \
    --env-vars \
        DATABASE_URL="postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@$POSTGRES_SERVER.postgres.database.azure.com:5432/mcp_db" \
        REDIS_URL="redis://:$REDIS_ACCESS_KEY@$REDIS_HOSTNAME:$REDIS_SSL_PORT" \
        SECRET_KEY="$(openssl rand -hex 32)" \
        API_HOST="0.0.0.0" \
        API_PORT="8000"

# Create Container App for Web Frontend
print_status "Creating Container App for Web Frontend: $WEB_APP_NAME"

az containerapp create \
    --name $WEB_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --environment $CONTAINER_APP_ENV \
    --image $ACR_LOGIN_SERVER/mcp-web:latest \
    --target-port 5173 \
    --ingress external \
    --registry-server $ACR_LOGIN_SERVER \
    --registry-username $ACR_USERNAME \
    --registry-password $ACR_PASSWORD \
    --env-vars \
        VITE_API_URL="https://$APP_NAME.$(az containerapp env show --name $CONTAINER_APP_ENV --resource-group $RESOURCE_GROUP --query 'properties.defaultDomain' --output tsv)"

# Get the URLs
BACKEND_URL=$(az containerapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query "properties.configuration.ingress.fqdn" --output tsv)
WEB_URL=$(az containerapp show --name $WEB_APP_NAME --resource-group $RESOURCE_GROUP --query "properties.configuration.ingress.fqdn" --output tsv)

print_success "MCP Stack deployed successfully to Azure!"
echo ""
echo "🌐 Access URLs:"
echo "   • Backend API: https://$BACKEND_URL"
echo "   • Web Dashboard: https://$WEB_URL"
echo "   • Health Check: https://$BACKEND_URL/api/v1/health"
echo ""
echo "📊 Azure Resources Created:"
echo "   • Resource Group: $RESOURCE_GROUP"
echo "   • Container Registry: $ACR_NAME"
echo "   • Container Apps Environment: $CONTAINER_APP_ENV"
echo "   • Backend API: $APP_NAME"
echo "   • Web Frontend: $WEB_APP_NAME"
echo "   • PostgreSQL Server: $POSTGRES_SERVER"
echo "   • Redis Cache: $REDIS_NAME"
echo ""
echo "🔑 Connection Details:"
echo "   • PostgreSQL: $POSTGRES_SERVER.postgres.database.azure.com:5432"
echo "   • Redis: $REDIS_HOSTNAME:$REDIS_SSL_PORT"
echo ""
echo "💡 Next Steps:"
echo "   1. Test the API: curl https://$BACKEND_URL/api/v1/health"
echo "   2. Visit the web dashboard: https://$WEB_URL"
echo "   3. Monitor resources in Azure Portal"
echo "   4. Scale up/down as needed"
echo ""
print_success "Deployment complete! Your MCP stack is now running in Azure!"
