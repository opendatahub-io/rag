from fastapi import FastAPI
from .routers import voice

app = FastAPI(title="Voice Processing API", description="API for voice transcription and synthesis")

# Include routers
app.include_router(voice.router)
