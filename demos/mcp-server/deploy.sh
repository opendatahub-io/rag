#!/bin/bash

# Copyright 2025 IBM, Red Hat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
