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

from fastapi import FastAPI
from .routers import voice

app = FastAPI(
    title="Voice Processing API",
    description="API for voice transcription and synthesis",
)


# Health check endpoint
@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "service": "Voice Processing API",
        "endpoints": {
            "transcribe": "POST /api/voice/transcribe",
            "complete": "POST /api/voice/complete",
            "speak": "POST /api/voice/speak",
            "session_start": "POST /api/voice/session/start",
            "chat": "POST /api/voice/chat",
            "conversation_clear": "POST /api/voice/conversation/clear",
        },
    }


# Include routers
app.include_router(voice.router)
