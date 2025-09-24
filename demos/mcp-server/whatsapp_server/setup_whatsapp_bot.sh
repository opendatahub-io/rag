#!/bin/bash

echo "🚀 Setting up WhatsApp RedBank Bot..."

# Create virtual environment if it doesn't exist
if [ ! -d "whatsapp_venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv whatsapp_venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source whatsapp_venv/bin/activate

# Install requirements
echo "📥 Installing dependencies..."
pip install -r whatsapp_requirements.txt

# Copy environment file if it doesn't exist
if [ ! -f "whatsapp_server/.env" ]; then
    echo "📝 Creating environment file..."
    cp whatsapp_server/env.example whatsapp_server/.env
    echo "⚠️  Please edit whatsapp_server/.env with your Evolution API credentials!"
fi

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit whatsapp_server/.env with your Evolution API credentials"
echo "2. Run: source whatsapp_venv/bin/activate"
echo "3. Run: python run_whatsapp_server.py"
echo ""
echo "Or use Docker:"
echo "1. Copy whatsapp_server/env.example to .env in project root"
echo "2. Edit .env with your credentials"
echo "3. Run: docker-compose -f docker-compose.whatsapp.yml up --build"
