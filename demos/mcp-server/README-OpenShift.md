# RedBank Financials MCP Server - OpenShift Deployment

This guide provides instructions for deploying the RedBank Financials MCP Server on OpenShift as a single container with embedded PostgreSQL.

## Architecture

The OpenShift deployment uses a single container approach with:
- **Embedded PostgreSQL**: Runs within the same container as the MCP server
- **Non-root user**: Runs as UID 1001 for OpenShift security compliance
- **Persistent storage**: Uses emptyDir volume for database data (can be upgraded to PVC)
- **Health checks**: Includes liveness and readiness probes
- **Route**: Exposes the service via OpenShift route with TLS

## Quick Deployment

### Option 1: Using the Build Script (Recommended)

1. **Run the build and deploy script:**
   ```bash
   chmod +x build-openshift.sh
   ./build-openshift.sh
   ```

2. **Follow the prompts to:**
   - Build the container image
   - Push to your registry (optional)
   - Deploy to OpenShift

### Option 2: Manual Deployment

1. **Build the image:**
   ```bash
   # Using Docker
   docker build -f Dockerfile.openshift -t redbank-mcp-server:latest .
   
   # Using Podman
   podman build -f Dockerfile.openshift -t redbank-mcp-server:latest .
   ```

2. **Push to registry:**
   ```bash
   # Tag for your registry
   docker tag redbank-mcp-server:latest quay.io/your-username/redbank-mcp-server:latest
   
   # Push to registry
   docker push quay.io/your-username/redbank-mcp-server:latest
   ```

3. **Update the deployment manifest:**
   ```bash
   # Edit openshift-deployment.yaml and update the image reference
   sed -i 's|image: redbank-mcp-server:latest|image: quay.io/your-username/redbank-mcp-server:latest|' openshift-deployment.yaml
   ```

4. **Deploy to OpenShift:**
   ```bash
   oc apply -f openshift-deployment.yaml
   ```

### Option 3: Using OpenShift BuildConfig (Source-to-Image)

1. **Update the git repository in the BuildConfig:**
   ```bash
   # Edit openshift-buildconfig.yaml with your git repository URL
   ```

2. **Create the BuildConfig:**
   ```bash
   oc apply -f openshift-buildconfig.yaml
   ```

3. **Start the build:**
   ```bash
   oc start-build redbank-mcp-server-build -n redbank-financials
   ```

4. **Deploy using the built image:**
   ```bash
   # Update deployment to use the ImageStream
   oc apply -f openshift-deployment.yaml
   ```

## Configuration

### Environment Variables
- `PYTHONUNBUFFERED=1`: Ensures Python output is not buffered
- `PGDATA=/opt/app-root/pgdata`: PostgreSQL data directory

### Resources
- **Requests**: 250m CPU, 512Mi memory
- **Limits**: 500m CPU, 1Gi memory
- **Storage**: emptyDir volume (ephemeral)

### Security
- Runs as non-root user (UID 1001)
- Uses OpenShift security context constraints
- TLS-enabled route with edge termination

## Accessing the Application

1. **Get the route URL:**
   ```bash
   oc get route redbank-mcp-route -n redbank-financials
   ```

2. **Access the MCP endpoint:**
   ```
   https://your-route-url/mcp
   ```

## Available MCP Tools

The deployed server provides these MCP tools:

1. **get_user_by_phone(phone_number: str)**
   - Get user details by phone number
   - Example: `"+1-555-123-4567"`

2. **get_statements(user_id: int)**
   - Get bank statements for a specific user

3. **get_transactions(statement_id: int)**
   - Get transactions for a specific statement

## Sample Data

Pre-populated with:
- 4 users with phone numbers
- 6 bank statements
- 30+ realistic transactions

Sample phone numbers:
- `+1-555-123-4567` (John Smith)
- `+1-555-987-6543` (Sarah Johnson)
- `+1-555-456-7890` (Michael Brown)
- `+1-555-321-0987` (Emily Davis)

## Management Commands

```bash
# View deployment status
oc get pods -n redbank-financials

# View logs
oc logs -f deployment/redbank-mcp-server -n redbank-financials

# Scale the application
oc scale deployment redbank-mcp-server --replicas=2 -n redbank-financials

# Get route information
oc get route -n redbank-financials

# Delete the application
oc delete -f openshift-deployment.yaml
```

## Troubleshooting

### Common Issues

1. **Pod not starting:**
   ```bash
   oc describe pod -l app=redbank-mcp-server -n redbank-financials
   ```

2. **Database initialization issues:**
   ```bash
   oc logs -f deployment/redbank-mcp-server -n redbank-financials
   ```

3. **Permission issues:**
   - Ensure the image runs as UID 1001
   - Check OpenShift security context constraints

4. **Route not accessible:**
   ```bash
   oc get route -n redbank-financials
   oc describe route redbank-mcp-route -n redbank-financials
   ```

### Health Checks

- **Readiness probe**: Checks `/mcp` endpoint after 30s
- **Liveness probe**: Checks `/mcp` endpoint after 60s
- **Startup time**: Allow up to 2 minutes for full initialization

## Production Considerations

For production deployments, consider:

1. **Persistent Storage**: Replace emptyDir with PVC for data persistence
2. **High Availability**: Use external PostgreSQL database
3. **Resource Limits**: Adjust based on load requirements
4. **Monitoring**: Add monitoring and alerting
5. **Backup**: Implement database backup strategy
6. **Security**: Use secrets for database credentials

## File Structure

```
├── Dockerfile.openshift          # OpenShift-compatible Dockerfile
├── openshift-deployment.yaml     # Deployment, Service, Route manifests
├── openshift-buildconfig.yaml    # BuildConfig for S2I builds
├── build-openshift.sh           # Build and deployment script
└── README-OpenShift.md          # This documentation
```
