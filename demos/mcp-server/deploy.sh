#!/bin/bash

echo "🏦 Starting RedBank Financials MCP Server Deployment"
echo "=================================================="

# Stop any existing containers
echo "Stopping existing containers..."
docker-compose down

# Build and start services
echo "Building and starting services..."
docker-compose up --build -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 10

# Check if services are running
echo "Checking service status..."
docker-compose ps

echo ""
echo "🎉 Deployment Complete!"
echo "=================================================="
echo "📊 PostgreSQL: Available on localhost:5432"
echo "🚀 MCP Server: Available on http://localhost:8000/mcp"
echo ""
echo "📋 Sample API calls:"
echo "  Get user by phone: POST /mcp with get_user_by_phone tool"
echo "  Get statements: POST /mcp with get_statements tool"  
echo "  Get transactions: POST /mcp with get_transactions tool"
echo ""
echo "To stop services: docker-compose down"
echo "To view logs: docker-compose logs -f"
