from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import base64
from ..services.whisper_svc import WhisperService
from ..services.agent_svc import ResponseService
from ..services.tts_svc import TTSService
from ..deps import get_logger
from ..config import settings

router = APIRouter(prefix="/api/voice", tags=["voice"])

_whisper = None
_response = None
_tts = None

def _get_whisper():
    global _whisper
    if _whisper is None:
        _whisper = WhisperService(settings.whisper_model, settings.whisper_url)
    return _whisper

def _get_response():
    global _response
    if _response is None:
        _response = ResponseService(settings.llamastack_url, settings.inference_model, settings.vector_store_name, settings.mcp_url)
    return _response

def _get_tts():
    global _tts
    if _tts is None:
        _tts = TTSService(settings.tts_url, settings.tts_route, settings.tts_voice)
    return _tts

@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...), logger = Depends(get_logger)):
    try:
        audio = await file.read()
        logger.info(f"Received audio file: {file.filename}, size: {len(audio)} bytes")
        text, dur = _get_whisper().transcribe(audio)
        logger.info(f"Transcribed {len(audio)} bytes to {len(text)} chars (dur≈{dur}s)")
        return {"text": text, "duration": dur}
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/complete")
async def complete(file: UploadFile = File(...), logger = Depends(get_logger)):
    audio = await file.read()
    text, _ = _get_whisper().transcribe(audio)

    tools_ctx = {"vdb_url": settings.vdb_url}
    agent_resp = _get_response().invoke(text, settings.model_instructions)
    response_text = agent_resp.get("output") or agent_resp.get("text") or str(agent_resp)

    wav = _get_tts().synthesize(response_text)

    return JSONResponse({
        "transcript": text,
        "agent_text": response_text,
        "wav_base64": base64.b64encode(wav).decode("ascii")
    })

@router.post("/speak")
async def speak(text: str):
    wav = _get_tts().synthesize(text)
    return StreamingResponse(iter([wav]), media_type="audio/wav")

@router.post("/session/start")
async def start_session(logger = Depends(get_logger)):
    """Start a new agent session"""
    try:
        session_id = _get_response().create_session()
        if session_id:
            logger.info(f"Created new agent session: {session_id}")
            return {"session_id": session_id, "status": "created"}
        else:
            raise HTTPException(status_code=500, detail="Failed to create agent session")
    except Exception as e:
        logger.error(f"Session creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
async def chat_with_agent(text: str, logger = Depends(get_logger)):
    """Chat with the agent using text input (for testing conversation continuity)"""
    try:
        tools_ctx = {"vdb_url": settings.vdb_url}
        agent_resp = _get_response().invoke(text, tools_ctx)
        agent_text = agent_resp.get("output") or agent_resp.get("text") or str(agent_resp)
        
        logger.info(f"User: {text}")
        logger.info(f"Agent: {agent_text}")
        
        return {
            "user_input": text,
            "agent_response": agent_text,
            "conversation_length": len(_get_response().conversation_history)
        }
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/conversation/clear")
async def clear_conversation(logger = Depends(get_logger)):
    """Clear the conversation history"""
    try:
        _get_response().clear_conversation()
        logger.info("Conversation history cleared")
        return {"status": "cleared", "message": "Conversation history has been cleared"}
    except Exception as e:
        logger.error(f"Clear conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
