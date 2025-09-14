#!/bin/bash

echo "🔧 Setting up SignLango deployment environment..."
echo ""

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "📦 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    echo "✅ Homebrew installed!"
else
    echo "✅ Homebrew already installed"
fi

# Install Docker Desktop
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker Desktop..."
    brew install --cask docker
    echo "✅ Docker Desktop installed!"
    echo "⚠️  Please start Docker Desktop from Applications folder"
else
    echo "✅ Docker already installed"
fi

# Install Azure CLI
if ! command -v az &> /dev/null; then
    echo "☁️  Installing Azure CLI..."
    brew install azure-cli
    echo "✅ Azure CLI installed!"
else
    echo "✅ Azure CLI already installed"
fi

echo ""
echo "🎉 Setup complete! Next steps:"
echo ""
echo "1. Start Docker Desktop (if not already running)"
echo "2. Login to Azure:"
echo "   az login"
echo ""
echo "3. Login to Azure Container Registry:"
echo "   az acr login --name signlangoacr"
echo ""
echo "4. Run the build and deploy script:"
echo "   ./build-and-deploy.sh"
echo ""
echo "5. Update App Services (after successful build):"
echo "   - Go to Azure Portal"
echo "   - Navigate to your App Services"
echo "   - Restart the services or trigger new deployment"

