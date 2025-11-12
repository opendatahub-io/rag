"""FastAPI server for WhatsApp webhook integration with Evolution API."""

import logging
import asyncio
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Dict, Any

from .config import settings
from .evolution_client import EvolutionAPIClient
from .models import WebhookData, MessageInfo
from .message_handler import MessageHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
evolution_client = EvolutionAPIClient()
message_handler = MessageHandler(evolution_client)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("Starting WhatsApp server...")
    
    try:
        # Test Evolution API connection
        await evolution_client.get_instance_info()
        logger.info("Successfully connected to Evolution API")
        
        # Get groups and find RedBank group
        groups_response = await evolution_client.get_groups()
        await message_handler.initialize_redbank_group(groups_response)
        
    except Exception as e:
        logger.error(f"Failed to initialize server: {str(e)}")
        # Continue anyway - we'll handle errors in the webhook
    
    yield
    
    logger.info("Shutting down WhatsApp server...")


# Create FastAPI app
app = FastAPI(
    title="WhatsApp RedBank Bot",
    description="FastAPI server that listens to WhatsApp messages and responds to RedBank group",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "WhatsApp RedBank Bot is running",
        "status": "healthy",
        "evolution_api_url": settings.evolution_api_url,
        "instance_id": settings.evolution_api_id
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    try:
        # Test Evolution API connection
        info = await evolution_client.get_instance_info()
        return {
            "status": "healthy",
            "evolution_api": "connected",
            "instance_info": info
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "evolution_api": "disconnected",
                "error": str(e)
            }
        )


@app.post("/webhook")
async def webhook_handler(request: Request, background_tasks: BackgroundTasks):
    """Handle incoming webhooks from Evolution API."""
    try:
        # Get raw body for logging
        body = await request.body()
        logger.info(f"Received webhook: {body.decode()}")
        
        # Parse webhook data
        webhook_data = WebhookData.parse_raw(body)
        
        # Handle the message in background
        background_tasks.add_task(
            message_handler.handle_webhook,
            webhook_data
        )
        
        return {"status": "received", "event": webhook_data.event}
        
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid webhook data: {str(e)}")


@app.get("/groups")
async def get_groups():
    """Get all groups for debugging."""
    try:
        groups = await evolution_client.get_groups()
        return groups
    except Exception as e:
        logger.error(f"Error fetching groups: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/test-message")
async def test_message(data: Dict[str, str]):
    """Test endpoint to send a message."""
    try:
        number = data.get("number")
        text = data.get("text", "Test message from RedBank bot")
        
        if not number:
            raise HTTPException(status_code=400, detail="Number is required")
        
        result = await evolution_client.send_text_message(number, text)
        return {"status": "sent", "result": result}
        
    except Exception as e:
        logger.error(f"Error sending test message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "whatsapp_server.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
