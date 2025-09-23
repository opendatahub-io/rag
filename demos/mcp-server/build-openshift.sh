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

# RedBank Financials MCP Server - OpenShift Build & Deploy Script

set -e

REGISTRY=${REGISTRY:-"quay.io"}
NAMESPACE=${NAMESPACE:-"your-username"}
IMAGE_NAME="redbank-mcp-server"
VERSION=${VERSION:-"latest"}
FULL_IMAGE_NAME="${REGISTRY}/${NAMESPACE}/${IMAGE_NAME}:${VERSION}"

echo "🏦 RedBank Financials MCP Server - OpenShift Deployment"
echo "======================================================="
echo "Registry: ${REGISTRY}"
echo "Namespace: ${NAMESPACE}"
echo "Image: ${FULL_IMAGE_NAME}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check required tools
echo "🔍 Checking required tools..."
if ! command_exists docker && ! command_exists podman; then
    echo "❌ Error: Neither docker nor podman found. Please install one of them."
    exit 1
fi

if ! command_exists oc; then
    echo "❌ Error: OpenShift CLI (oc) not found. Please install it."
    exit 1
fi

# Use podman if available, otherwise docker
if command_exists podman; then
    CONTAINER_CMD="podman"
else
    CONTAINER_CMD="docker"
fi

echo "✅ Using ${CONTAINER_CMD} for container operations"

# Build the image
echo ""
echo "🔨 Building OpenShift-compatible image..."
${CONTAINER_CMD} build -f Dockerfile.openshift -t ${IMAGE_NAME}:${VERSION} .
${CONTAINER_CMD} tag ${IMAGE_NAME}:${VERSION} ${FULL_IMAGE_NAME}

echo "✅ Image built successfully: ${FULL_IMAGE_NAME}"

# Push to registry (optional)
read -p "🚀 Push image to registry? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📤 Pushing image to registry..."
    ${CONTAINER_CMD} push ${FULL_IMAGE_NAME}
    echo "✅ Image pushed successfully"
    
    # Update deployment with correct image
    sed -i.bak "s|image: redbank-mcp-server:latest|image: ${FULL_IMAGE_NAME}|g" openshift-deployment.yaml
    echo "✅ Updated deployment manifest with image: ${FULL_IMAGE_NAME}"
fi

# Deploy to OpenShift (optional)
echo ""
read -p "🎯 Deploy to OpenShift? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Deploying to OpenShift..."
    
    # Check if logged in to OpenShift
    if ! oc whoami >/dev/null 2>&1; then
        echo "❌ Error: Not logged in to OpenShift. Please run 'oc login' first."
        exit 1
    fi
    
    # Apply the deployment
    oc apply -f openshift-deployment.yaml
    
    echo "✅ Deployment applied successfully"
    echo ""
    echo "📊 Checking deployment status..."
    oc get pods -n redbank-financials
    
    echo ""
    echo "🌐 Getting route URL..."
    sleep 5
    ROUTE_URL=$(oc get route redbank-mcp-route -n redbank-financials -o jsonpath='{.spec.host}' 2>/dev/null || echo "Route not ready yet")
    if [ "$ROUTE_URL" != "Route not ready yet" ]; then
        echo "🎉 Application will be available at: https://${ROUTE_URL}/mcp"
    else
        echo "⏳ Route not ready yet. Check with: oc get route -n redbank-financials"
    fi
fi

echo ""
echo "📋 Useful OpenShift Commands:"
echo "  View pods:        oc get pods -n redbank-financials"
echo "  View logs:        oc logs -f deployment/redbank-mcp-server -n redbank-financials"
echo "  Get route:        oc get route -n redbank-financials"
echo "  Scale app:        oc scale deployment redbank-mcp-server --replicas=2 -n redbank-financials"
echo "  Delete app:       oc delete -f openshift-deployment.yaml"
echo ""
echo "🏦 RedBank Financials MCP Server deployment complete!"
