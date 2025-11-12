"""Message handling logic for WhatsApp webhooks."""

import logging
from typing import Dict, Any, Optional, List
from .evolution_client import EvolutionAPIClient
from .models import WebhookData, MessageInfo, GroupInfo
from .config import settings

logger = logging.getLogger(__name__)


class MessageHandler:
    """Handles incoming WhatsApp messages and responses."""
    
    def __init__(self, evolution_client: EvolutionAPIClient):
        self.evolution_client = evolution_client
        self.redbank_group_jid: Optional[str] = None
        self.redbank_group_info: Optional[GroupInfo] = None
    
    async def initialize_redbank_group(self, groups_response: Dict[str, Any]):
        """Find and store RedBank group information."""
        try:
            groups_data = groups_response.get("data", [])
            if isinstance(groups_data, list):
                for group_data in groups_data:
                    group_subject = group_data.get("subject", "").lower()
                    if settings.redbank_group_name.lower() in group_subject:
                        self.redbank_group_jid = group_data.get("id")
                        self.redbank_group_info = GroupInfo(**group_data)
                        logger.info(f"Found RedBank group: {self.redbank_group_jid}")
                        break
            
            if not self.redbank_group_jid:
                logger.warning(f"RedBank group not found in {len(groups_data)} groups")
                # Log available groups for debugging
                for group_data in groups_data:
                    logger.info(f"Available group: {group_data.get('subject', 'Unknown')}")
                    
        except Exception as e:
            logger.error(f"Error initializing RedBank group: {str(e)}")
    
    async def handle_webhook(self, webhook_data: WebhookData):
        """Process incoming webhook data."""
        try:
            event = webhook_data.event
            logger.info(f"Processing event: {event}")
            
            # Handle different event types
            if event == "MESSAGES_UPSERT":
                await self._handle_message_upsert(webhook_data.data)
            elif event == "MESSAGES_SET":
                await self._handle_messages_set(webhook_data.data)
            else:
                logger.debug(f"Ignoring event type: {event}")
                
        except Exception as e:
            logger.error(f"Error handling webhook: {str(e)}")
    
    async def _handle_message_upsert(self, data: Dict[str, Any]):
        """Handle MESSAGES_UPSERT event (new messages)."""
        try:
            messages = data.get("messages", [])
            
            for message_data in messages:
                await self._process_message(message_data)
                
        except Exception as e:
            logger.error(f"Error handling message upsert: {str(e)}")
    
    async def _handle_messages_set(self, data: Dict[str, Any]):
        """Handle MESSAGES_SET event (message history)."""
        # Usually we don't want to respond to historical messages
        logger.debug("Ignoring MESSAGES_SET event (historical messages)")
    
    async def _process_message(self, message_data: Dict[str, Any]):
        """Process a single message."""
        try:
            # Parse message info
            key = message_data.get("key", {})
            remote_jid = key.get("remoteJid", "")
            from_me = key.get("fromMe", False)
            message_content = message_data.get("message", {})
            push_name = message_data.get("pushName", "Unknown")
            
            logger.info(f"Processing message from {push_name} ({remote_jid})")
            
            # Skip messages from ourselves
            if from_me:
                logger.debug("Skipping message from self")
                return
            
            # Check if message is from RedBank group
            if not self._is_redbank_group_message(remote_jid):
                logger.debug(f"Message not from RedBank group: {remote_jid}")
                return
            
            # Extract message text
            message_text = self._extract_message_text(message_content)
            if not message_text:
                logger.debug("No text content found in message")
                return
            
            logger.info(f"RedBank group message from {push_name}: {message_text}")
            
            # Send welcome response
            await self._send_welcome_response(remote_jid)
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
    
    def _is_redbank_group_message(self, remote_jid: str) -> bool:
        """Check if message is from RedBank group."""
        if not self.redbank_group_jid:
            # If we haven't found the group yet, check by name pattern
            return (
                remote_jid.endswith("@g.us") and  # Is a group
                settings.redbank_group_name.lower() in remote_jid.lower()
            )
        
        return remote_jid == self.redbank_group_jid
    
    def _extract_message_text(self, message_content: Dict[str, Any]) -> Optional[str]:
        """Extract text from message content."""
        # Handle different message types
        if "conversation" in message_content:
            return message_content["conversation"]
        
        if "extendedTextMessage" in message_content:
            extended = message_content["extendedTextMessage"]
            return extended.get("text")
        
        if "imageMessage" in message_content:
            image = message_content["imageMessage"]
            return image.get("caption")
        
        if "videoMessage" in message_content:
            video = message_content["videoMessage"]
            return video.get("caption")
        
        if "documentMessage" in message_content:
            document = message_content["documentMessage"]
            return document.get("caption")
        
        # Add more message types as needed
        logger.debug(f"Unknown message type: {list(message_content.keys())}")
        return None
    
    async def _send_welcome_response(self, group_jid: str):
        """Send welcome message to the group."""
        try:
            await self.evolution_client.send_text_message(
                number=group_jid,
                text=settings.welcome_message
            )
            logger.info(f"Sent welcome message to {group_jid}")
            
        except Exception as e:
            logger.error(f"Error sending welcome message: {str(e)}")
    
    def get_redbank_group_info(self) -> Dict[str, Any]:
        """Get current RedBank group information."""
        return {
            "group_jid": self.redbank_group_jid,
            "group_info": self.redbank_group_info.dict() if self.redbank_group_info else None
        }
