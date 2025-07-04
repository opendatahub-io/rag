# Installation of llama-stack-milvus distribution on OpenShift

## Prerequisite
* The Llama Stack Operator installed on your cluster following this [guide](https://github.com/opendatahub-io/llama-stack-k8s-operator?tab=readme-ov-file#installation)

These `kustomize` manifests provides a way to install a LlamaStackDistribution
that can be used to perform Retrieval-Augmented Generation (RAG) with Milvus as the vector database.


## Installation
Several installation `overlays` are provided depending on which model you want to use and which
vLLM serving strategy you want to adopt.
See [overlay folder](./overlays) to find out which model are available.

## KServe vLLM + llama 3.2

This is the default and the most convenient way for OpenShift AI as it uses the existing KServer template to run
vLLM and serve the model.
To install the `LlamaStackDistribution` that uses vLLM backed by KServe that serves `llama3.2` run:
```
oc apply -k base/
```

## vLLM standalone serving granite 3.3

This is the easiest way as it uses `granite` is a public model on HuggingFace.
To install the `LlamaStackDistribution` using `granite3.3` run:
```
oc apply -k overlays/vllm-standalone-granite3.3
```

## vLLM standalone serving llama 3.2
To be able to use `llama 3.x` models you need to get a `HuggingFace` token that has access to the
`llama` models. So, you need to have the appropriate permissions for that. And then:

```
kubectl apply -k overlays/vllm-standalone-llama3.2
oc patch secret hf-token-secret --type='merge' -p='{"data":{"HF_TOKEN":"'$(echo -n "hf_your_token" | base64)'"}}'
```

