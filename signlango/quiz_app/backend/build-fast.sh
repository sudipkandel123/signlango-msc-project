#!/bin/bash

# Fast build script for backend
set -e

echo "🚀 Starting fast backend build..."

# Get ACR login server
ACR_LOGIN_SERVER="signlangoacr.azurecr.io"

# Build with buildx for better caching
docker buildx build \
    --platform linux/amd64 \
    --cache-from type=registry,ref=$ACR_LOGIN_SERVER/signlango-backend:cache \
    --cache-to type=registry,ref=$ACR_LOGIN_SERVER/signlango-backend:cache,mode=max \
    --tag signlango-backend:latest \
    --tag $ACR_LOGIN_SERVER/signlango-backend:latest \
    --push \
    .

echo "✅ Backend build completed and pushed to ACR!"
