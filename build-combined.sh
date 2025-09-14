#!/bin/bash

# Combined build script for SignLango full-stack application
set -e

echo "🚀 Starting SignLango combined build..."

# Get ACR login server
ACR_LOGIN_SERVER="signlangoacr.azurecr.io"

# Build the combined image locally first
echo "🔨 Building combined image locally..."
docker build --tag signlango-combined:latest .

# Test the local build
echo "🧪 Testing local build..."
docker run --rm -d --name signlango-combined-test -p 80:80 signlango-combined:latest
sleep 10

if docker ps | grep -q signlango-combined-test; then
    echo "✅ Local build test successful!"
    
    # Test the endpoints
    echo "🔍 Testing endpoints..."
    if curl -f http://localhost/health > /dev/null 2>&1; then
        echo "✅ Backend health check passed!"
    else
        echo "❌ Backend health check failed!"
    fi
    
    if curl -f http://localhost/ > /dev/null 2>&1; then
        echo "✅ Frontend serving passed!"
    else
        echo "❌ Frontend serving failed!"
    fi
    
    docker stop signlango-combined-test
else
    echo "❌ Local build test failed!"
    docker logs signlango-combined-test
    docker stop signlango-combined-test
    exit 1
fi

# Tag and push to ACR
echo "📤 Tagging and pushing to ACR..."
docker tag signlango-combined:latest $ACR_LOGIN_SERVER/signlango-combined:latest
docker push $ACR_LOGIN_SERVER/signlango-combined:latest

echo "✅ Combined build completed and pushed to ACR!"
echo ""
echo "🔄 To update your App Service:"
echo "1. Go to Azure Portal"
echo "2. Navigate to your App Service"
echo "3. Update container image to: signlangoacr.azurecr.io/signlango-combined:latest"
echo "4. Restart the service"
