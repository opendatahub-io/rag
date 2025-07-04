#!/bin/zsh
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

# Extracts the manifests from an existing installed llamastack+llamastack-distribution

oc eksporter secret hf-token-secret --drop data.HF_TOKEN  > base/secret.yaml
oc eksporter configmap llama3-chat-template               > base/config-map.yaml
oc eksporter deployment vllm                              > base/deployment.yaml
oc eksporter svc vllm --drop spec.clusterIPs              > base/service.yaml
oc eksporter route vllm --drop spec.host                  > base/route.yaml
oc eksporter LlamaStackDistribution                       > base/llama-stack-distribution.yaml
