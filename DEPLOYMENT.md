# RAG Stack Deployment Guide

This guide will help you deploy and test the RAG (Retrieval Augmented Generation) stack on OpenShift.

## Prerequisites

1. OpenShift CLI (`oc`) installed and configured
2. Access to an OpenShift cluster
3. Basic understanding of Kubernetes/OpenShift concepts

## Deployment Steps

### 1. Login to OpenShift

```bash
oc login --token=<your-token> --server=<your-cluster-url>
```

### 2. Create a New Project (Optional)

```bash
oc new-project rag-stack
```

### 3. Deploy the Stack

The project offers three deployment options:

#### Option A: Default Setup (KServe vLLM + Llama 3.2)
```bash
oc apply -k stack/base/
```

#### Option B: vLLM standalone serving Granite 3.3
```bash
oc apply -k stack/overlays/vllm-standalone-granite3.3
```

#### Option C: vLLM standalone serving Llama 3.2
```bash
oc apply -k stack/overlays/vllm-standalone-llama3.2
oc patch secret hf-token-secret --type='merge' -p='{"data":{"HF_TOKEN":"'$(echo -n "hf_your_token" | base64)'"}}'
```

### 4. Verify Deployment

Check if all pods are running:
```bash
oc get pods
```

Expected output should show:
- `lsd-llama-milvus-*` pod
- `vllm-predictor-*` pod

Both pods should be in `Running` state.

## Testing the Deployment

### 1. Port Forward the Services

Before starting port-forward, check if ports are already in use:
```bash
lsof -i :8080
lsof -i :8321
```

If the ports are in use, you can either:
- Kill the existing process: `kill <PID>`
- Or use different ports

#### Port Forward vLLM Predictor
```bash
oc port-forward svc/vllm-predictor 8080:80
```

#### Port Forward Llama Stack Service
```bash
oc port-forward svc/lsd-llama-milvus-service 8321:8321
```

### 2. Test the APIs

#### Direct vLLM Access (Port 8080)

##### Check Available Models
```bash
curl http://localhost:8080/v1/models
```

##### Send a Test Query
```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vllm",
    "prompt": "What is Retrieval Augmented Generation (RAG)?",
    "max_tokens": 100
  }'
```

#### Get Route URL
To get the route URL for external access:
```bash
# Get the route URL
ROUTE_URL=$(oc get route lsd-llama-milvus -o jsonpath='{.spec.host}')
echo "Route URL: http://$ROUTE_URL"
```

#### Llama Stack Access (Port 8321)

##### Health Check
```bash
# Using port-forward
curl http://localhost:8321/v1/health

# Using route
curl http://$ROUTE_URL/v1/health
```

##### List Available Models
```bash
# Using port-forward
curl http://localhost:8321/v1/models

# Using route
curl http://$ROUTE_URL/v1/models
```

##### OpenAI-Compatible Completions
```bash
# Using port-forward
curl -X POST http://localhost:8321/v1/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vllm",
    "messages": [
      {
        "role": "user",
        "content": "What is Retrieval Augmented Generation (RAG)?"
      }
    ]
  }'

# Using route
curl -X POST http://$ROUTE_URL/v1/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vllm",
    "messages": [
      {
        "role": "user",
        "content": "What is Retrieval Augmented Generation (RAG)?"
      }
    ]
  }'
```

### 3. API Parameters

#### vLLM Direct Access
Required parameters for the completion endpoint:
- `model`: "vllm" (the model name)
- `prompt`: Your question or text
- `max_tokens`: Maximum length of the response (e.g., 100)

#### Llama Stack OpenAI-Compatible API
Required parameters for the chat completions endpoint:
- `model`: "vllm" (the model name)
- `messages`: Array of message objects with:
  - `role`: "user" or "assistant"
  - `content`: The message text

### 4. Available Models

The deployment provides access to two models:
- `vllm`: Language model for text generation
- `ibm-granite/granite-embedding-125m-english`: Embedding model for vector operations

## Troubleshooting

### Common Issues

1. **Pods not starting**
   ```bash
   oc get pods
   oc describe pod <pod-name>
   oc logs <pod-name>
   ```

2. **Service not responding**
   - Verify port-forward is running
   - Check if pods are in Running state
   - Verify service endpoints:
     ```bash
     oc get svc
     oc describe svc vllm-predictor
     ```

3. **Model not found**
   - Verify model name using `/v1/models` endpoint
   - Check model configuration in deployment

4. **Port conflicts**
   - Check for existing port-forward processes:
     ```bash
     lsof -i :8080
     ```
   - Kill existing process if needed:
     ```bash
     kill <PID>
     ```
   - Or use a different port:
     ```bash
     oc port-forward svc/vllm-predictor 8081:80
     ```

5. **Connection issues**
   - Ensure port-forward is running in a separate terminal
   - Check if the service is accessible:
     ```bash
     curl http://localhost:8080/v1/health
     ```
   - Verify network policies if applicable

## Cleanup

To completely remove the project and all its resources from OpenShift, follow these steps:

1. Delete the entire project:
   ```bash
   oc delete project rag-stack
   ```

2. Verify that the project has been completely removed:
   ```bash
   oc get project rag-stack
   ```
   You should see an error message indicating that the namespace was not found, confirming successful deletion.

3. If you had any port-forward processes running, they will be automatically terminated when the project is deleted. However, if you need to manually check for and kill any remaining port-forward processes:
   ```bash
   # Check for processes using port 8080
   lsof -i :8080

   # Kill the process if found (replace PID with the actual process ID)
   kill <PID>
   ```

After completing these steps, all resources associated with the RAG stack will be completely removed from your OpenShift cluster.

## Additional Resources

- [OpenShift Documentation](https://docs.openshift.com/)
- [KServe Documentation](https://kserve.github.io/website/)
- [vLLM Documentation](https://vllm.readthedocs.io/)