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

import os


class Settings:
    whisper_model = os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo-quantized")
    llamastack_url = os.getenv("LLAMASTACK_URL", "http://localhost:8321")
    agent_route = "/v1/agents"
    agent_name = (
        "6fdd1315-2326-402e-966f-9ee7ff0c8b30"  # Using the agent ID from remote LSD
    )
    agent_api_key = "key"
    tts_url = os.getenv("TTS_URL", "http://localhost:8001")
    tts_route = "/tts"
    tts_voice = "default"
    vdb_url = os.getenv("VDB_URL", "http://localhost:8002")
    inference_model = os.getenv("INFERENCE_MODEL", "vllm-inference/qwen3-14b-awq")
    vector_store_name = os.getenv("VECTOR_STORE_NAME", "redbank-kb-vector-store")
    whisper_url = os.getenv("WHISPER_URL", "http://localhost:80/v1")
    mcp_url = os.getenv("MCP_URL", "http://redbank-mcp-server:8000/mcp")
    model_instructions = os.getenv("MODEL_INSTRUCTIONS", "")


settings = Settings()
