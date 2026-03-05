.PHONY: deploy-all delete-all deploy-qwen3-14b-awq delete-qwen3-14b-awq \
deploy-lsd-server delete-lsd-server deploy-configs delete-configs \
deploy-postgres delete-postgres deploy-minio delete-minio \
deploy-dspa delete-dspa

NAMESPACE := ragas-evaluation
KUBECTL := kubectl

ifneq ($(shell which oc),)
	KUBECTL := oc
endif

deploy-all:
	$(KUBECTL) create namespace $(NAMESPACE) || true
	$(MAKE) deploy-configs
	$(MAKE) deploy-minio
	$(MAKE) deploy-postgres
	$(MAKE) deploy-dspa
	$(MAKE) deploy-qwen3-14b-awq
	$(MAKE) deploy-gpt-oss-20b
	$(KUBECTL) wait --for=condition=available --timeout=1200s deployment --all -n $(NAMESPACE) || true
	$(MAKE) deploy-lsd-server
	$(KUBECTL) wait --for=condition=available --timeout=300s deployment --all -n $(NAMESPACE) || true

delete-all: delete-lsd-server delete-qwen3-14b-awq delete-gpt-oss-20b delete-dspa delete-postgres delete-minio delete-configs

deploy-configs:
	$(KUBECTL) apply -f ../shared-resources/aws-credentials.yaml -n $(NAMESPACE)
	$(KUBECTL) apply -f ../shared-resources/kubeflow-ragas-config.yaml -n $(NAMESPACE)
	$(KUBECTL) apply -f deployment-yamls/llama-stack-config.yaml -n $(NAMESPACE)
	$(KUBECTL) apply -f deployment-yamls/inference-models-secret.yaml -n $(NAMESPACE)

delete-configs:
	$(KUBECTL) delete -f deployment-yamls/llama-stack-config.yaml -n $(NAMESPACE) || true
	$(KUBECTL) delete -f ../shared-resources/kubeflow-ragas-config.yaml -n $(NAMESPACE) || true
	$(KUBECTL) delete -f ../shared-resources/aws-credentials.yaml -n $(NAMESPACE) || true
	$(KUBECTL) delete -f deployment-yamls/inference-models-secret.yaml -n $(NAMESPACE) || true

deploy-qwen3-14b-awq:
	$(KUBECTL) apply -f ../../common-deployments/qwen3-14b-awq-deployment.yaml -n $(NAMESPACE)

delete-qwen3-14b-awq:
	$(KUBECTL) delete -f ../../common-deployments/qwen3-14b-awq-deployment.yaml -n $(NAMESPACE) || true

deploy-gpt-oss-20b:
	$(KUBECTL) apply -f ../../common-deployments/gpt-oss-20b.yaml -n $(NAMESPACE)

delete-gpt-oss-20b:
	$(KUBECTL) delete -f ../../common-deployments/gpt-oss-20b.yaml -n $(NAMESPACE) || true

deploy-lsd-server:
	$(KUBECTL) apply -f deployment-yamls/llama-stack-distribution.yaml -n $(NAMESPACE)

delete-lsd-server:
	$(KUBECTL) delete -f deployment-yamls/llama-stack-distribution.yaml -n $(NAMESPACE) || true

deploy-postgres:
	$(KUBECTL) apply -f ../../common-deployments/postgres-deployment.yaml -n $(NAMESPACE)

delete-postgres:
	$(KUBECTL) delete -f ../../common-deployments/postgres-deployment.yaml -n $(NAMESPACE) || true

deploy-minio:
	$(KUBECTL) apply -f ../../common-deployments/minio-deployment.yaml -n $(NAMESPACE)

delete-minio:
	$(KUBECTL) delete -f ../../common-deployments/minio-deployment.yaml -n $(NAMESPACE) || true

deploy-dspa:
	$(KUBECTL) apply -f ../shared-resources/dspa-deployment.yaml -n $(NAMESPACE)
	$(KUBECTL) apply -f ../shared-resources/llama-stack-network-policy.yaml -n $(NAMESPACE)

delete-dspa:
	$(KUBECTL) delete -f ../shared-resources/llama-stack-network-policy.yaml -n $(NAMESPACE) || true
	$(KUBECTL) delete -f ../shared-resources/dspa-deployment.yaml -n $(NAMESPACE) || true
