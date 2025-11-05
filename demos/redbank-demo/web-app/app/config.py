import os

class Settings:
    whisper_model = "base"
    whisper_device = "cpu"
    whisper_compute_type = "int8"
    llamastack_url = os.getenv("LLAMASTACK_URL", "http://localhost:8321")
    agent_route = "/v1/agents"
    agent_name = "6fdd1315-2326-402e-966f-9ee7ff0c8b30"  # Using the agent ID from remote LSD
    agent_api_key = "key"
    tts_url = os.getenv("TTS_URL", "http://localhost:8001")
    tts_route = "/tts"
    tts_voice = "default"
    vdb_url = os.getenv("VDB_URL", "http://localhost:8002")
    inference_model = os.getenv('INFERENCE_MODEL', 'llama3.2:3b')

settings = Settings()
