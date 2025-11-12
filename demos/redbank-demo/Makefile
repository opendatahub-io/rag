.PHONY: deploy-postgres-mcp deploy-postgres delete-postgres-mcp delete-postgres deploy-mcp delete-mcp \
deploy-all delete-all deploy-whisper delete-whisper deploy-kokoro delete-kokoro \
deploy-qwen3-14b-awq delete-qwen3-14b-awq \
deploy-lsd-server delete-lsd-server \
deploy-voice-api delete-voice-api deploy-chat-ui delete-chat-ui

NAMESPACE := redbank-demo
KUBECTL := kubectl

ifneq ($(shell which oc),)
	KUBECTL := oc
endif

############ Deploy/Delete All ############
deploy-all:
	$(KUBECTL) create namespace $(NAMESPACE) || true
	$(MAKE) -j deploy-qwen3-14b-awq deploy-whisper deploy-kokoro deploy-postgres-mcp deploy-voice-api deploy-chat-ui
	$(KUBECTL) wait --for=condition=available --timeout=1200s deployment --all -n $(NAMESPACE)
	$(MAKE) deploy-lsd-server
	$(KUBECTL) wait --for=condition=available --timeout=300s deployment --all -n $(NAMESPACE)
delete-all: delete-qwen3-14b-awq delete-whisper delete-kokoro delete-postgres-mcp delete-lsd-server delete-voice-api delete-chat-ui

############ PostgreSQL and MCP Server ############
deploy-postgres-mcp: deploy-postgres deploy-mcp
delete-postgres-mcp: delete-mcp delete-postgres

deploy-postgres:
	$(KUBECTL) apply -k postgres-db/ -n $(NAMESPACE)
delete-postgres:
	$(KUBECTL) delete -k postgres-db/ -n $(NAMESPACE)

deploy-mcp:
	cd mcp-server && ./deploy.sh
delete-mcp:
	$(KUBECTL) delete -f mcp-server/mcp-server.yaml -n $(NAMESPACE) || true
	$(KUBECTL) delete route redbank-mcp-server -n $(NAMESPACE) || true
	$(KUBECTL) delete build build-redbank-mcp-server -n $(NAMESPACE) || true
	$(KUBECTL) delete buildconfig build-redbank-mcp-server -n $(NAMESPACE) || true

############ Whisper STT ############
deploy-whisper:
	$(KUBECTL) apply -f deployment-yamls/whisper-deployment.yaml -n $(NAMESPACE)
delete-whisper:
	$(KUBECTL) delete -f deployment-yamls/whisper-deployment.yaml -n $(NAMESPACE) || true

############ Kokoro TTS ############
deploy-kokoro:
	$(KUBECTL) create sa kokoro-sa -n $(NAMESPACE) || true
	$(KUBECTL) adm policy add-scc-to-user anyuid -z kokoro-sa -n $(NAMESPACE) || true
	$(KUBECTL) apply -f deployment-yamls/kokoro-deployment.yaml -n $(NAMESPACE)
delete-kokoro:
	$(KUBECTL) delete -f deployment-yamls/kokoro-deployment.yaml -n $(NAMESPACE) || true
	$(KUBECTL) delete sa kokoro-sa -n $(NAMESPACE) || true

############ Qwen3-14B-AWQ AI Model ############
deploy-qwen3-14b-awq:
	$(KUBECTL) apply -f deployment-yamls/qwen3-14b-awq-deployment.yaml -n $(NAMESPACE)
delete-qwen3-14b-awq:
	$(KUBECTL) delete -f deployment-yamls/qwen3-14b-awq-deployment.yaml -n $(NAMESPACE) || true

############ LlamaStackDistribution ############
deploy-lsd-server:
	$(KUBECTL) apply -f deployment-yamls/llamastackdistribution.yaml -n $(NAMESPACE)
delete-lsd-server:
	$(KUBECTL) delete -f deployment-yamls/llamastackdistribution.yaml -n $(NAMESPACE) || true

############ Voice API Server ############
deploy-voice-api:
	cd chat-bot/voice-api-server && ./build_and_deploy.sh
	$(KUBECTL) apply -f deployment-yamls/voice-api-server.yaml -n $(NAMESPACE)
delete-voice-api:
	$(KUBECTL) delete -f deployment-yamls/voice-api-server.yaml -n $(NAMESPACE) || true
	$(KUBECTL) delete build build-redbank-voice-api-server -n $(NAMESPACE) || true
	$(KUBECTL) delete buildconfig build-redbank-voice-api-server -n $(NAMESPACE) || true

############ Chat UI ############
deploy-chat-ui:
	cd chat-bot/ui && ./build_and_deploy.sh
	$(KUBECTL) apply -f deployment-yamls/chat-ui.yaml -n $(NAMESPACE)
delete-chat-ui:
	$(KUBECTL) delete -f deployment-yamls/chat-ui.yaml -n $(NAMESPACE) || true
	$(KUBECTL) delete build build-redbank-chat-ui -n $(NAMESPACE) || true
	$(KUBECTL) delete buildconfig build-redbank-chat-ui -n $(NAMESPACE) || true
