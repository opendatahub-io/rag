"""Evolution API client for WhatsApp operations."""

import httpx
import logging
from typing import Dict, Any, Optional
from .config import settings

logger = logging.getLogger(__name__)


class EvolutionAPIClient:
    """Client for interacting with Evolution API."""
    
    def __init__(self):
        self.base_url = settings.evolution_api_url
        self.instance_id = settings.evolution_api_id
        self.token = settings.evolution_api_token
        self.headers = {
            "apikey": self.token,
            "Content-Type": "application/json"
        }
    
    async def send_text_message(self, number: str, text: str) -> Dict[str, Any]:
        """Send a text message to a WhatsApp number or group.
        
        Args:
            number: Phone number or group JID
            text: Message text to send
            
        Returns:
            API response as dictionary
        """
        url = f"{self.base_url}/message/sendText/{self.instance_id}"
        
        payload = {
            "number": number,
            "text": text
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"Message sent successfully to {number}")
                return result
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error sending message: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            raise
    
    async def get_instance_info(self) -> Dict[str, Any]:
        """Get instance information."""
        url = f"{self.base_url}/instance/fetchInstances"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            logger.error(f"Error fetching instance info: {str(e)}")
            raise
    
    async def get_groups(self) -> Dict[str, Any]:
        """Fetch all groups for the instance."""
        url = f"{self.base_url}/group/fetchAllGroups/{self.instance_id}"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            logger.error(f"Error fetching groups: {str(e)}")
            raise
