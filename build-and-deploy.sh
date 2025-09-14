#!/bin/bash

# Main build and deploy script for SignLango
set -e

echo "🚀 Starting SignLango build and deploy process..."

# Configuration
ACR_LOGIN_SERVER="signlangoacr.azurecr.io"
BACKEND_DIR="signlango/quiz_app/backend"
FRONTEND_DIR="signlango/quiz_app/frontend"

# Login to Azure Container Registry (you'll need to run this manually first)
# az acr login --name signlangoacr

echo "📦 Building and pushing backend..."
cd $BACKEND_DIR
chmod +x build-fast.sh
./build-fast.sh

echo "📦 Building and pushing frontend..."
cd ../../../
cd $FRONTEND_DIR
chmod +x build-fast.sh
./build-fast.sh

echo "✅ All builds completed successfully!"
echo ""
echo "🔄 To update your App Services, you can:"
echo "1. Go to Azure Portal"
echo "2. Navigate to your App Services:"
echo "   - signlango-backend.azurewebsites.net"
echo "   - signlango-frontend.azurewebsites.net"
echo "3. Go to 'Deployment Center'"
echo "4. Trigger a new deployment or restart the service"
echo ""
echo "🔗 Or use Azure CLI:"
echo "az webapp restart --name signlango-backend --resource-group <your-resource-group>"
echo "az webapp restart --name signlango-frontend --resource-group <your-resource-group>"

