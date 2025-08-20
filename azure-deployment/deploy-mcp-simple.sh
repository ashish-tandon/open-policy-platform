#!/bin/bash

# 🚀 Simple Azure MCP Stack Deployment
# This script deploys your MCP stack to Azure Container Instances (ACI)

set -e

echo "🚀 Simple Azure MCP Stack Deployment"
echo "===================================="
echo ""

# Check Azure CLI
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI not found. Please install it first."
    exit 1
fi

# Login to Azure
echo "🔐 Logging into Azure..."
az login

# Configuration
RESOURCE_GROUP="mcp-stack-rg"
LOCATION="eastus"
POSTGRES_SERVER="mcp-postgres-$(date +%s | tail -c 5)"
POSTGRES_USER="mcpadmin"
POSTGRES_PASSWORD="MCP$(date +%s | tail -c 8)!"

echo "📋 Creating Azure resources..."

# Create Resource Group
echo "📦 Creating resource group: $RESOURCE_GROUP"
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create PostgreSQL Flexible Server
echo "🗄️ Creating PostgreSQL server: $POSTGRES_SERVER"
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

# Create database
echo "📊 Creating database..."
az postgres flexible-server db create \
    --resource-group $RESOURCE_GROUP \
    --server-name $POSTGRES_SERVER \
    --database-name mcp_db

# Create Redis Cache
echo "🔴 Creating Redis Cache..."
REDIS_NAME="mcp-redis-$(date +%s | tail -c 5)"

az redis create \
    --resource-group $RESOURCE_GROUP \
    --name $REDIS_NAME \
    --location $LOCATION \
    --sku Basic \
    --vm-size C0

# Get Redis details
REDIS_HOSTNAME=$(az redis show --name $REDIS_NAME --resource-group $RESOURCE_GROUP --query "hostName" --output tsv)
REDIS_SSL_PORT=$(az redis show --name $REDIS_NAME --resource-group $RESOURCE_GROUP --query "sslPort" --output tsv)
REDIS_ACCESS_KEY=$(az redis list-keys --name $REDIS_NAME --resource-group $RESOURCE_GROUP --query "primaryKey" --output tsv)

echo "✅ Redis created: $REDIS_HOSTNAME"

# Deploy Backend API using Azure Container Instances
echo "🚀 Deploying Backend API..."

az container create \
    --resource-group $RESOURCE_GROUP \
    --name mcp-backend-api \
    --image python:3.11-slim \
    --ports 8000 \
    --dns-name-label mcp-backend-$(date +%s | tail -c 5) \
    --environment-variables \
        DATABASE_URL="postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@$POSTGRES_SERVER.postgres.database.azure.com:5432/mcp_db" \
        REDIS_URL="redis://:$REDIS_ACCESS_KEY@$REDIS_HOSTNAME:$REDIS_SSL_PORT" \
        SECRET_KEY="$(openssl rand -hex 32)" \
        API_HOST="0.0.0.0" \
        API_PORT="8000" \
        ENVIRONMENT="production" \
    --command-line "bash -c 'apt-get update && apt-get install -y git && git clone https://github.com/ashish-tandon/open-policy-platform.git && cd open-policy-platform/backend && pip install -r requirements.txt && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000'" \
    --cpu 2 \
    --memory 4

# Deploy Web Frontend using Azure Container Instances
echo "🌐 Deploying Web Frontend..."

az container create \
    --resource-group $RESOURCE_GROUP \
    --name mcp-web-frontend \
    --image node:18-alpine \
    --ports 5173 \
    --dns-name-label mcp-web-$(date +%s | tail -c 5) \
    --environment-variables \
        VITE_API_URL="http://mcp-backend-$(date +%s | tail -c 5).$LOCATION.azurecontainer.io:8000" \
        NODE_ENV="production" \
    --command-line "sh -c 'apk add --no-cache git && git clone https://github.com/ashish-tandon/open-policy-platform.git && cd open-policy-platform/web && npm install && npm run build && npm run preview -- --host 0.0.0.0 --port 5173'" \
    --cpu 1 \
    --memory 2

# Get the URLs
BACKEND_URL=$(az container show --name mcp-backend-api --resource-group $RESOURCE_GROUP --query "ipAddress.fqdn" --output tsv)
WEB_URL=$(az container show --name mcp-web-frontend --resource-group $RESOURCE_GROUP --query "ipAddress.fqdn" --output tsv)

echo ""
echo "🎉 MCP Stack deployed successfully to Azure!"
echo ""
echo "🌐 Access URLs:"
echo "   • Backend API: http://$BACKEND_URL:8000"
echo "   • Web Dashboard: http://$WEB_URL:5173"
echo "   • Health Check: http://$BACKEND_URL:8000/api/v1/health"
echo ""
echo "📊 Azure Resources Created:"
echo "   • Resource Group: $RESOURCE_GROUP"
echo "   • Backend API: mcp-backend-api"
echo "   • Web Frontend: mcp-web-frontend"
echo "   • PostgreSQL Server: $POSTGRES_SERVER"
echo "   • Redis Cache: $REDIS_NAME"
echo ""
echo "🔑 Connection Details:"
echo "   • PostgreSQL: $POSTGRES_SERVER.postgres.database.azure.com:5432"
echo "   • Redis: $REDIS_HOSTNAME:$REDIS_SSL_PORT"
echo ""
echo "💡 Next Steps:"
echo "   1. Test the API: curl http://$BACKEND_URL:8000/api/v1/health"
echo "   2. Visit the web dashboard: http://$WEB_URL:5173"
echo "   3. Monitor resources in Azure Portal"
echo "   4. Scale up/down as needed"
echo ""
echo "✅ Deployment complete! Your MCP stack is now running in Azure!"
