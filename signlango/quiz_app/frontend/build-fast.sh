#!/bin/bash

# Fast build script for frontend
set -e

echo "🚀 Starting fast frontend build..."

# Get ACR login server
ACR_LOGIN_SERVER="signlangoacr.azurecr.io"

# Build locally first
echo "🔨 Building frontend image locally..."
docker build --tag signlango-frontend:latest .

# Test the local build
echo "🧪 Testing local build..."
docker run --rm -d --name signlango-frontend-test -p 3000:80 signlango-frontend:latest
sleep 5
if docker ps | grep -q signlango-frontend-test; then
    echo "✅ Local build test successful!"
    docker stop signlango-frontend-test
else
    echo "❌ Local build test failed!"
    docker logs signlango-frontend-test
    docker stop signlango-frontend-test
    exit 1
fi

# Tag and push to ACR
echo "📤 Tagging and pushing to ACR..."
docker tag signlango-frontend:latest $ACR_LOGIN_SERVER/signlango-frontend:latest
docker push $ACR_LOGIN_SERVER/signlango-frontend:latest

echo "✅ Frontend build completed and pushed to ACR!"

