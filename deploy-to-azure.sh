#!/bin/bash

# 🚀 Azure MCP Stack Deployment Launcher
# This script helps you choose and run the Azure deployment

echo "🚀 Azure MCP Stack Deployment Launcher"
echo "======================================"
echo ""
echo "Choose your deployment method:"
echo ""
echo "1️⃣  Simple Deployment (Container Instances) - Easiest, ~$365/month"
echo "2️⃣  Production Deployment (Container Apps) - Scalable, ~$550/month"
echo "3️⃣  View Deployment Guide"
echo "4️⃣  Exit"
echo ""

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting Simple Azure Deployment..."
        echo "This will create:"
        echo "   • Resource Group"
        echo "   • PostgreSQL Database"
        echo "   • Redis Cache"
        echo "   • Backend API Container"
        echo "   • Web Frontend Container"
        echo ""
        read -p "Continue? (y/n): " confirm
        if [[ $confirm == "y" || $confirm == "Y" ]]; then
            ./azure-deployment/deploy-mcp-simple.sh
        else
            echo "Deployment cancelled."
        fi
        ;;
    2)
        echo ""
        echo "🚀 Starting Production Azure Deployment..."
        echo "This will create:"
        echo "   • Resource Group"
        echo "   • Azure Container Registry"
        echo "   • Container Apps Environment"
        echo "   • PostgreSQL Database"
        echo "   • Redis Cache"
        echo "   • Scalable Backend API"
        echo "   • Scalable Web Frontend"
        echo ""
        read -p "Continue? (y/n): " confirm
        if [[ $confirm == "y" || $confirm == "Y" ]]; then
            ./azure-deployment/azure-deploy-mcp-stack.sh
        else
            echo "Deployment cancelled."
        fi
        ;;
    3)
        echo ""
        echo "📖 Opening Azure Deployment Guide..."
        echo ""
        echo "📋 Prerequisites:"
        echo "   • Azure Account with active subscription"
        echo "   • Azure CLI installed (already done!)"
        echo "   • Git access to your repository"
        echo ""
        echo "🚀 Quick Start:"
        echo "   1. Run: az login"
        echo "   2. Choose deployment method above"
        echo "   3. Wait for deployment to complete"
        echo "   4. Access your MCP stack via provided URLs"
        echo ""
        echo "📖 Full Guide: azure-deployment/AZURE_DEPLOYMENT_GUIDE.md"
        echo ""
        read -p "Press Enter to continue..."
        ;;
    4)
        echo "Goodbye! 👋"
        exit 0
        ;;
    *)
        echo "Invalid choice. Please run the script again."
        exit 1
        ;;
esac
