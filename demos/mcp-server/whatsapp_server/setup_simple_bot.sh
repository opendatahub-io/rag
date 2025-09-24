#!/bin/bash

echo "🤖 Setting up Simple RedBank WhatsApp Bot..."

# Create virtual environment if it doesn't exist
if [ ! -d "bot_venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv bot_venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source bot_venv/bin/activate

# Install requirements
echo "📥 Installing dependencies..."
pip install -r bot_requirements.txt

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
echo "2. Run: source bot_venv/bin/activate"
echo "3. Run: python redbank_bot.py"
echo ""
echo "The bot will:"
echo "- 🔍 Find the RedBank group automatically"
echo "- 📨 Poll for new messages every 10 seconds"
echo "- 💬 Reply with 'Welcome to RedBank' to any new message"
echo "- 📝 Log all activity to redbank_bot.log"
