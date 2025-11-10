import os

class Settings:
    whisper_model = os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo-quantized")
    llamastack_url = os.getenv("LLAMASTACK_URL", "http://localhost:8321")
    agent_route = "/v1/agents"
    agent_name = "6fdd1315-2326-402e-966f-9ee7ff0c8b30"  # Using the agent ID from remote LSD
    agent_api_key = "key"
    tts_url = os.getenv("TTS_URL", "http://localhost:8001")
    tts_route = "/tts"
    tts_voice = "default"
    vdb_url = os.getenv("VDB_URL", "http://localhost:8002")
    inference_model = os.getenv('INFERENCE_MODEL', 'vllm-inference/qwen2-5')
    vector_store_name = os.getenv('VECTOR_STORE_NAME', 'redbank-kb-vector-store')
    whisper_url = os.getenv("WHISPER_URL", "http://localhost:80/v1")

settings = Settings()
