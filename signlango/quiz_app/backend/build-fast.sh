#!/bin/bash

# Fast build script for backend
set -e

echo "🚀 Starting fast backend build..."

# Get ACR login server
ACR_LOGIN_SERVER="signlangoacr.azurecr.io"

# Build locally first
echo "🔨 Building backend image locally..."
docker build --tag signlango-backend:latest .

# Test the local build
echo "🧪 Testing local build..."
docker run --rm -d --name signlango-backend-test -p 8000:8000 signlango-backend:latest
sleep 5
if docker ps | grep -q signlango-backend-test; then
    echo "✅ Local build test successful!"
    docker stop signlango-backend-test
else
    echo "❌ Local build test failed!"
    docker logs signlango-backend-test
    docker stop signlango-backend-test
    exit 1
fi

# Tag and push to ACR
echo "📤 Tagging and pushing to ACR..."
docker tag signlango-backend:latest $ACR_LOGIN_SERVER/signlango-backend:latest
docker push $ACR_LOGIN_SERVER/signlango-backend:latest

echo "✅ Backend build completed and pushed to ACR!"
