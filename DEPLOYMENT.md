# SignLango Deployment Guide

This guide will help you build and deploy the SignLango application to Azure App Services using Azure Container Registry.

## Prerequisites

Before starting the deployment process, you need to have the following tools installed:

1. **Docker Desktop** - For building container images
2. **Azure CLI** - For Azure authentication and management
3. **Homebrew** - For installing tools on macOS

## Quick Setup

Run the setup script to install all required tools:

```bash
./setup-deployment.sh
```

## Manual Setup (if needed)

### 1. Install Docker Desktop

```bash
brew install --cask docker
```

### 2. Install Azure CLI

```bash
brew install azure-cli
```

### 3. Start Docker Desktop

Open Docker Desktop from your Applications folder and wait for it to start.

## Authentication

### 1. Login to Azure

```bash
az login
```

This will open a browser window for you to authenticate with your Azure account.

### 2. Login to Azure Container Registry

```bash
az acr login --name signlangoacr
```

## Project Structure

```
signlango-msc-project/
├── signlango/quiz_app/
│   ├── backend/
│   │   ├── Dockerfile          # Backend container configuration
│   │   ├── build-fast.sh       # Backend build script
│   │   ├── requirements.txt    # Python dependencies
│   │   └── run.py             # Main backend application
│   └── frontend/
│       ├── Dockerfile          # Frontend container configuration
│       ├── build-fast.sh       # Frontend build script
│       ├── nginx.conf          # Nginx configuration
│       └── package.json        # Node.js dependencies
├── build-and-deploy.sh         # Main deployment script
└── setup-deployment.sh         # Environment setup script
```

## Building and Deploying

### Option 1: Automated Build (Recommended)d 


Run the main build script to build and push both frontend and backend:

```bash
./build-and-deploy.sh
```

### Option 2: Manual Build

#### Build Backend

```bash
cd signlango/quiz_app/backend
./build-fast.sh
```

#### Build Frontend

```bash
cd signlango/quiz_app/frontend
./build-fast.sh
```

## Azure Resources

Your application uses the following Azure resources:

- **Container Registry**: `signlangoacr.azurecr.io`

  - Repository: `signlango-backend`
  - Repository: `signlango-frontend`
- **App Services**:

  - Backend: `signlango-backend.azurewebsites.net`
  - Frontend: `signlango-frontend.azurewebsites.net`

## Updating App Services

After successfully building and pushing the images to ACR, you need to update your App Services:

### Option 1: Azure Portal

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to your App Services:
   - `signlango-backend`
   - `signlango-frontend`
3. Go to **Deployment Center**
4. Click **Sync** or **Restart** to pull the latest images

### Option 2: Azure CLI

```bash
# Restart backend app service
az webapp restart --name signlango-backend --resource-group <your-resource-group>

# Restart frontend app service
az webapp restart --name signlango-frontend --resource-group <your-resource-group>
```

## Troubleshooting

### Common Issues

1. **Docker not running**

   - Make sure Docker Desktop is started
   - Check Docker status: `docker info`
2. **Azure authentication issues**

   - Re-login: `az login`
   - Check subscription: `az account show`
3. **ACR login issues**

   - Re-login to ACR: `az acr login --name signlangoacr`
   - Check ACR access: `az acr repository list --name signlangoacr`
4. **Build failures**

   - Check Docker logs: `docker logs <container-id>`
   - Verify Dockerfile syntax
   - Check for missing dependencies

### Useful Commands

```bash
# Check Docker status
docker info

# List local images
docker images

# Check Azure CLI status
az account show

# List ACR repositories
az acr repository list --name signlangoacr

# View app service logs
az webapp log tail --name signlango-backend --resource-group <your-resource-group>
```

## Environment Variables

Make sure your App Services have the necessary environment variables configured:

### Backend Environment Variables

- `GOOGLE_API_KEY` - For Google Generative AI
- `SECRET_KEY` - For JWT token signing
- `DATABASE_URL` - If using a database

### Frontend Environment Variables

- `REACT_APP_API_URL` - Backend API URL

## Monitoring

After deployment, monitor your applications:

1. **Application Logs**: Check Azure Portal > App Service > Log stream
2. **Performance**: Use Azure Application Insights
3. **Health Checks**: Monitor the `/health` endpoint on backend

## Rollback

If you need to rollback to a previous version:

1. Go to Azure Portal > App Service > Deployment Center
2. Select a previous deployment
3. Click **Redeploy**

Or use Azure CLI:

```bash
az webapp deployment source config --name <app-name> --resource-group <resource-group> --repo-url <previous-image-tag>
```
