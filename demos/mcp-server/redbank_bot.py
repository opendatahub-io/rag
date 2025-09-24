#!/usr/bin/env python3
"""
Simple WhatsApp RedBank Bot - Polls for messages and auto-replies
"""

import os
import time
import json
import requests
import logging
import base64
import tempfile
from datetime import datetime, timedelta
from typing import Set, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv("whatsapp_server/.env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("redbank_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RedBankBot:
    """Simple WhatsApp bot that polls for RedBank group messages and replies."""
    
    def __init__(self):
        # Evolution API configuration
        self.api_url = os.getenv("EVOLUTION_API_URL", "https://api.evoapicloud.com")
        self.instance_id = os.getenv("EVOLUTION_API_ID")
        self.api_token = os.getenv("EVOLUTION_API_TOKEN")
        
        if not all([self.instance_id, self.api_token]):
            raise ValueError("Missing Evolution API credentials in environment variables")
        
        # Bot configuration
        self.group_name = os.getenv("REDBANK_GROUP_NAME", "RedBank").lower()
        self.welcome_message = os.getenv("WELCOME_MESSAGE", "Welcome to RedBank")
        self.poll_interval = int(os.getenv("POLL_INTERVAL", "10"))  # seconds
        
        # Headers for API requests
        self.headers = {
            "apikey": self.api_token,
            "Content-Type": "application/json"
        }
        
        # Track processed messages to avoid duplicates
        self.processed_messages: Set[str] = set()
        self.redbank_group_jid: Optional[str] = None
        self.last_poll_time: int = 0
        
        logger.info(f"🤖 RedBank Bot initialized")
        logger.info(f"📱 Instance ID: {self.instance_id}")
        logger.info(f"🎯 Target Group: {self.group_name}")
        logger.info(f"💬 Welcome Message: {self.welcome_message}")
        logger.info(f"⏱️  Poll Interval: {self.poll_interval}s")
    
    def find_redbank_group(self) -> bool:
        """Find the RedBank group JID."""
        try:
            url = f"{self.api_url}/group/fetchAllGroups/{self.instance_id}?getParticipants=false"
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            groups_data = response.json()
            groups = groups_data if isinstance(groups_data, list) else groups_data.get("data", [])
            
            for group in groups:
                group_subject = group.get("subject", "").lower()
                if self.group_name in group_subject:
                    self.redbank_group_jid = group.get("id")
                    logger.info(f"✅ Found RedBank group: {group.get('subject')} ({self.redbank_group_jid})")
                    return True
            
            logger.warning(f"❌ RedBank group not found among {len(groups)} groups")
            logger.info("Available groups:")
            for group in groups[:5]:  # Show first 5 groups
                logger.info(f"  - {group.get('subject', 'Unknown')}")
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error finding RedBank group: {str(e)}")
            return False
    
    def get_recent_messages(self) -> list:
        """Get messages from RedBank group since last poll."""
        if not self.redbank_group_jid:
            return []
        
        try:
            url = f"{self.api_url}/chat/findMessages/{self.instance_id}"
            
            payload = {
                "where": {
                    "key": {
                        "remoteJid": self.redbank_group_jid
                    }
                },
                "limit": 50  # Get more messages to ensure we don't miss any
            }
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            messages_data = response.json()
            # Fix: API returns messages.records, not data
            messages = messages_data.get("messages", {}).get("records", [])
            
            # Filter for messages since last poll
            new_messages = []
            current_time = int(datetime.now().timestamp())
            
            # If this is the first poll, set last_poll_time to 5 minutes ago to avoid old messages
            if self.last_poll_time == 0:
                self.last_poll_time = current_time - 300  # 5 minutes ago
            
            for message in messages:
                timestamp = message.get("messageTimestamp", 0)
                # Get messages since last poll
                if timestamp > self.last_poll_time:
                    new_messages.append(message)
            
            # Update last poll time
            self.last_poll_time = current_time
            
            # Sort by timestamp (oldest first) so we process messages in order
            new_messages.sort(key=lambda x: x.get("messageTimestamp", 0))
            
            return new_messages
            
        except Exception as e:
            logger.error(f"❌ Error getting messages: {str(e)}")
            return []
    
    def extract_message_text(self, message_content: Dict[str, Any]) -> Optional[str]:
        """Extract text from message content."""
        if "conversation" in message_content:
            return message_content["conversation"]
        
        if "extendedTextMessage" in message_content:
            return message_content["extendedTextMessage"].get("text")
        
        if "imageMessage" in message_content:
            return message_content["imageMessage"].get("caption")
        
        if "videoMessage" in message_content:
            return message_content["videoMessage"].get("caption")
        
        if "documentMessage" in message_content:
            return message_content["documentMessage"].get("caption")
        
        # Voice/Audio messages
        if "audioMessage" in message_content:
            audio_msg = message_content["audioMessage"]
            duration = audio_msg.get("seconds", 0)
            is_voice_note = audio_msg.get("ptt", False)
            msg_type = "Voice Note" if is_voice_note else "Audio"
            return f"[{msg_type} - {duration}s]"
        
        # Add more message types as needed
        logger.debug(f"Unknown message type: {list(message_content.keys())}")
        return None
    
    def is_voice_message(self, message_content: Dict[str, Any]) -> bool:
        """Check if message is a voice/audio message."""
        return "audioMessage" in message_content
    
    def get_voice_message_info(self, message_content: Dict[str, Any]) -> Dict[str, Any]:
        """Extract voice message information."""
        if not self.is_voice_message(message_content):
            return {}
        
        audio_msg = message_content["audioMessage"]
        return {
            "url": audio_msg.get("url", ""),
            "mimetype": audio_msg.get("mimetype", ""),
            "fileLength": audio_msg.get("fileLength", 0),
            "seconds": audio_msg.get("seconds", 0),
            "ptt": audio_msg.get("ptt", False),  # Push-to-talk (voice note)
            "mediaKey": audio_msg.get("mediaKey", ""),
            "fileEncSha256": audio_msg.get("fileEncSha256", "")
        }
    
    def download_voice_message(self, message_data: Dict[str, Any]) -> Optional[str]:
        """Download voice message and return base64 encoded audio."""
        try:
            # Get the message ID to use with the Evolution API
            message_key = message_data.get("key", {})
            message_id = message_key.get("id")
            
            if not message_id:
                logger.error("❌ No message ID found for voice download")
                return None
            
            # Use Evolution API to get base64 of media message
            url = f"{self.api_url}/chat/getBase64FromMediaMessage/{self.instance_id}"
            
            payload = {
                "message": {
                    "key": {
                        "id": message_id
                    }
                },
                "convertToMp4": False  # Keep as audio format
            }
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            base64_audio = result.get("base64", "")
            
            if base64_audio:
                logger.info(f"✅ Downloaded voice message: {message_id}")
                return base64_audio
            else:
                logger.error(f"❌ No base64 data in response for message: {message_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error downloading voice message: {str(e)}")
            return None
    
    def send_voice_message(self, base64_audio: str, mimetype: str = "audio/ogg; codecs=opus") -> bool:
        """Send voice message to RedBank group."""
        if not self.redbank_group_jid:
            return False
        
        try:
            url = f"{self.api_url}/message/sendWhatsAppAudio/{self.instance_id}"
            
            payload = {
                "number": self.redbank_group_jid,
                "audio": base64_audio
            }
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            logger.info(f"✅ Sent voice message to RedBank group")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error sending voice message: {str(e)}")
            return False
    
    def send_custom_message(self, message_text: str) -> bool:
        """Send custom message to RedBank group."""
        if not self.redbank_group_jid:
            return False
        
        try:
            url = f"{self.api_url}/message/sendText/{self.instance_id}"
            
            payload = {
                "number": self.redbank_group_jid,
                "text": message_text
            }
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            logger.info(f"✅ Sent custom message to RedBank group: {message_text[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error sending custom message: {str(e)}")
            return False
    
    def send_audio_url_message(self) -> bool:
        """Send a pre-recorded audio message to RedBank group."""
        if not self.redbank_group_jid:
            return False
        
        try:
            # Use a simple public audio file for testing
            url = f"{self.api_url}/message/sendMedia/{self.instance_id}"
            
            payload = {
                "number": self.redbank_group_jid,
                "mediatype": "audio",
                "media": "https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav",
                "fileName": "welcome_redbank.wav"
            }
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            logger.info(f"✅ Sent audio message to RedBank group")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error sending audio message: {str(e)}")
            return False
    
    def send_welcome_message(self) -> bool:
        """Send welcome message to RedBank group."""
        return self.send_custom_message(self.welcome_message)
    
    def display_messages(self, messages):
        """Display messages in a readable format."""
        if not messages:
            logger.info("📭 No recent messages found in RedBank group")
            return
        
        logger.info(f"📬 Found {len(messages)} recent messages in RedBank group:")
        print("\n" + "="*60)
        print("📨 REDBANK GROUP MESSAGES")
        print("="*60)
        
        for i, message in enumerate(messages, 1):
            key = message.get("key", {})
            message_id = key.get("id", "unknown")
            from_me = key.get("fromMe", False)
            push_name = message.get("pushName", "Unknown")
            timestamp = message.get("messageTimestamp", 0)
            message_content = message.get("message", {})
            
            # Convert timestamp to readable format
            if timestamp:
                import datetime
                dt = datetime.datetime.fromtimestamp(timestamp)
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                time_str = "Unknown time"
            
            # Extract message text
            text = self.extract_message_text(message_content)
            participant_name = key.get("participantPushName", push_name)
            
            print(f"\n📝 Message {i}:")
            print(f"   👤 From: {participant_name}")
            print(f"   🕐 Time: {time_str}")
            print(f"   🤖 From Me: {from_me}")
            print(f"   💬 Text: {text or 'No text content'}")
            print(f"   🆔 ID: {message_id}")
        
        print("="*60 + "\n")
    
    def process_messages(self):
        """Process messages since last poll and reply to new ones."""
        messages = self.get_recent_messages()
        
        if not messages:
            logger.debug("🔍 No new messages since last poll")
            return
        
        # Display messages for debugging
        self.display_messages(messages)
        
        new_messages_count = 0
        
        for message in messages:
            try:
                # Get message details
                key = message.get("key", {})
                message_id = key.get("id")
                from_me = key.get("fromMe", False)
                push_name = key.get("participantPushName", message.get("pushName", "Unknown"))
                message_content = message.get("message", {})
                
                # Skip if already processed
                if message_id in self.processed_messages:
                    continue
                
                # Skip messages from ourselves (bot messages)
                # Check both fromMe flag and participant field
                participant = key.get("participant", "")
                
                # Log message details for debugging
                logger.debug(f"📋 Message details: ID={message_id}, fromMe={from_me}, participant={participant}, pushName={push_name}")
                
                if from_me:
                    logger.info(f"⏭️  Skipping own message: {message_id}")
                    self.processed_messages.add(message_id)
                    continue
                
                # Extract message text
                text = self.extract_message_text(message_content)
                
                if text:
                    timestamp = message.get("messageTimestamp", 0)
                    dt = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()
                    time_str = dt.strftime("%H:%M:%S")
                    
                    logger.info(f"🆕 NEW MESSAGE at {time_str} from {push_name}: {text}")
                    new_messages_count += 1
                    
                    # Check if it's a voice message
                    if self.is_voice_message(message_content):
                        logger.info(f"🎤 Voice message detected from {push_name}")
                        
                        # Download the voice message
                        voice_data = self.download_voice_message(message)
                        if voice_data:
                            # Reply with the same voice message
                            success = self.send_voice_message(voice_data)
                            if success:
                                logger.info(f"🎤 Replied with voice message to {push_name}")
                            else:
                                logger.error(f"❌ Failed to send voice reply to {push_name}")
                        else:
                            logger.error(f"❌ Failed to download voice message from {push_name}")
                            # Try to send a pre-recorded audio reply
                            logger.info(f"🎤 Trying audio URL reply to {push_name}")
                            success = self.send_audio_url_message()
                            if success:
                                logger.info(f"🎤 Sent audio reply to {push_name}")
                            else:
                                # Final fallback to text reply
                                success = self.send_welcome_message()
                                if success:
                                    logger.info(f"📝 Sent text reply instead to {push_name}")
                    else:
                        # Regular text message - send welcome reply
                        success = self.send_welcome_message()
                        if success:
                            logger.info(f"✅ Replied to {push_name}")
                        else:
                            logger.error(f"❌ Failed to reply to {push_name}")
                    
                    # Add small delay to avoid rate limiting
                    time.sleep(3)
                
                # Mark as processed
                self.processed_messages.add(message_id)
                
            except Exception as e:
                logger.error(f"❌ Error processing message: {str(e)}")
        
        if new_messages_count > 0:
            logger.info(f"📊 Processed {new_messages_count} new messages")
        
        # Clean up old processed messages (keep last 1000)
        if len(self.processed_messages) > 1000:
            self.processed_messages = set(list(self.processed_messages)[-500:])
    
    def run(self):
        """Main bot loop."""
        logger.info("🚀 Starting RedBank Bot...")
        
        # Find RedBank group first
        if not self.find_redbank_group():
            logger.error("❌ Cannot start bot without RedBank group")
            return
        
        logger.info(f"🔄 Starting polling loop (every {self.poll_interval}s)")
        
        try:
            while True:
                try:
                    self.process_messages()
                    time.sleep(self.poll_interval)
                    
                except KeyboardInterrupt:
                    logger.info("👋 Bot stopped by user")
                    break
                    
                except Exception as e:
                    logger.error(f"❌ Error in main loop: {str(e)}")
                    logger.info(f"⏳ Waiting {self.poll_interval}s before retry...")
                    time.sleep(self.poll_interval)
                    
        except Exception as e:
            logger.error(f"❌ Fatal error: {str(e)}")


def main():
    """Main entry point."""
    print("🤖 RedBank WhatsApp Bot")
    print("=" * 30)
    
    # Check environment file
    env_file = "whatsapp_server/.env"
    if not os.path.exists(env_file):
        print(f"❌ Environment file not found: {env_file}")
        print(f"📝 Please copy whatsapp_server/env.example to {env_file}")
        print("   and configure your Evolution API credentials")
        return 1
    
    try:
        bot = RedBankBot()
        bot.run()
        return 0
        
    except ValueError as e:
        logger.error(f"❌ Configuration error: {str(e)}")
        print("\n📝 Please check your environment configuration:")
        print("   - EVOLUTION_API_URL")
        print("   - EVOLUTION_API_ID") 
        print("   - EVOLUTION_API_TOKEN")
        return 1
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
