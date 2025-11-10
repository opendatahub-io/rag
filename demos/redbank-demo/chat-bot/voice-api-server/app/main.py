from fastapi import FastAPI
from .routers import voice

app = FastAPI(title="Voice Processing API", description="API for voice transcription and synthesis")

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
            "conversation_clear": "POST /api/voice/conversation/clear"
        }
    }

# Include routers
app.include_router(voice.router)
