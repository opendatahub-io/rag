#!/usr/bin/env python3
"""
Direct RedBank WhatsApp Bot for Ignas
Listens for direct messages from Ignas only and replies with AI banking assistance
"""

import os
import time
import json
import requests
import logging
from datetime import datetime
from typing import Set, Dict, Any, Optional
from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger

def extract_response_text(response):
    """Extract the actual text response from the LlamaStack response object"""
    try:
        # The response is a generator, so we need to consume it
        if hasattr(response, '__iter__'):
            # Look for the final response in the turn_complete event
            final_response = None
            
            for chunk in response:
                if hasattr(chunk, 'event') and chunk.event:
                    if hasattr(chunk.event, 'payload') and chunk.event.payload:
                        payload = chunk.event.payload
                        
                        # Check if this is the turn_complete event with the final response
                        if (hasattr(payload, 'event_type') and 
                            payload.event_type == 'turn_complete' and
                            hasattr(payload, 'turn') and payload.turn and
                            hasattr(payload.turn, 'output_message') and payload.turn.output_message and
                            hasattr(payload.turn.output_message, 'content')):
                            
                            final_response = payload.turn.output_message.content
                            break
            
            return final_response if final_response else "No final response found"
        else:
            return str(response)
    except Exception as e:
        import traceback
        return f"Error extracting response: {e}\nTraceback: {traceback.format_exc()}"

# Load environment variables
load_dotenv(".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ignas_direct_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IgnasDirectBot:
    """Direct WhatsApp bot for Ignas only."""
    
    def __init__(self):
        # Evolution API configuration
        self.api_url = os.getenv("EVOLUTION_API_URL", "https://api.evoapicloud.com")
        self.instance_id = os.getenv("EVOLUTION_API_ID")
        self.api_token = os.getenv("EVOLUTION_API_TOKEN")
        
        if not all([self.instance_id, self.api_token]):
            raise ValueError("Missing Evolution API credentials")
        
        # Headers for API requests
        self.headers = {
            "apikey": self.api_token,
            "Content-Type": "application/json"
        }
        
        # Ignas specific configuration
        self.ignas_phone = "353851480072"  # Ignas's phone number
        self.ignas_jid = f"{self.ignas_phone}@s.whatsapp.net"  # Direct message JID
        
        # Track processed message IDs
        self.processed_message_ids = set()
        # Start with timestamp from 1 hour ago to catch recent messages
        one_hour_ago = int(datetime.now().timestamp()) - 3600
        self.last_processed_timestamp = one_hour_ago
        
        # Initialize LlamaStack
        self.llamastack_url = os.getenv("LLAMASTACK_URL", "http://ragathon-team-1-ragathon-team-1.apps.llama-rag-pool-b84hp.aws.rh-ods.com")
        self.client = None
        self.agent = None
        self.session_id = None
        
        logger.info(f"🤖 Ignas Direct Bot initialized")
        logger.info(f"📱 Target: Ignas ({self.ignas_jid})")
        logger.info(f"🕐 Processing messages newer than: {datetime.fromtimestamp(one_hour_ago)}")
    
    def setup_ai_agent(self):
        """Setup LlamaStack AI agent."""
        try:
            self.client = LlamaStackClient(base_url=self.llamastack_url)
            
            self.agent = Agent(
                self.client,
                model="vllm-inference/llama-4-scout-17b-16e-w4a16",
                instructions="""You are a banking assistant for Ignas Baranauskas. His phone number is +353 85 148 0072.

When Ignas asks for banking information, use the MCP banking tools:
1. get_user_by_phone(phone_number="+353 85 148 0072", session_id="ignas-session")
2. get_statements(user_id=<user_id_from_step1>, session_id="ignas-session") 
3. get_transactions(statement_id=<statement_id>, session_id="ignas-session")

Only call tools if the user specifically asks for banking information. Be friendly and personal.""",
                tools=[
                    "mcp::redbank-financials", 
                    {
                        "name": "builtin::rag/knowledge_search",
                        "args": {"vector_db_ids": ['vs_1f1dd1b7-49ad-4ceb-8e8d-f0bf9afe2179']},
                    }
                ], 
            )
            
            self.session_id = self.agent.create_session("ignas-direct-bot")
            logger.info("✅ AI agent ready for Ignas")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error setting up AI agent: {e}")
            return False
    
    def get_ignas_messages(self):
        """Get unprocessed messages from Ignas."""
        try:
            url = f"{self.api_url}/chat/findMessages/{self.instance_id}"
            
            payload = {
                "where": {
                    "key": {
                        "remoteJid": self.ignas_jid  # Direct messages from Ignas
                    }
                }
            }
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            messages = response.json().get("messages", {}).get("records", [])
            
            logger.info(f"📋 Retrieved {len(messages)} messages from Ignas")
            
            if not messages:
                return []
            
            # Sort messages by timestamp (newest first) 
            sorted_messages = sorted(messages, key=lambda x: x.get("messageTimestamp", 0), reverse=True)
            
            unprocessed_messages = []
            
            # Check all messages for unprocessed ones
            for message in sorted_messages:
                key = message.get("key", {})
                message_id = key.get("id")
                from_me = key.get("fromMe", False)
                timestamp = message.get("messageTimestamp", 0)
                
                # Skip bot messages (messages from us)
                if from_me:
                    continue
                
                # Skip already processed messages
                if message_id in self.processed_message_ids:
                    continue
                
                # Skip old messages
                if timestamp <= self.last_processed_timestamp:
                    continue
                
                # Extract message text
                msg_content = message.get("message", {})
                text = msg_content.get("conversation") or msg_content.get("extendedTextMessage", {}).get("text", "")
                
                if text and text.strip():
                    dt = datetime.fromtimestamp(timestamp) if timestamp else "Unknown"
                    logger.info(f"📝 Found unprocessed message from Ignas: {dt} | text='{text[:50]}...' | id={message_id}")
                    
                    unprocessed_messages.append({
                        "id": message_id,
                        "text": text.strip(),
                        "phone": f"+353 85 148 0072",  # Ignas's formatted phone
                        "name": "Ignas",
                        "timestamp": timestamp
                    })
            
            return unprocessed_messages
            
        except Exception as e:
            logger.error(f"❌ Error getting Ignas messages: {e}")
            return []
    
    def get_ai_response(self, message_text, sender_name, sender_phone):
        """Get response from AI agent."""
        try:
            user_message = f"Ignas is asking: {message_text}"
            
            response = self.agent.create_turn(
                messages=[{"role": "user", "content": user_message}],
                session_id=self.session_id
            )
            
            # Extract response using the proven method
            logger.info("📡 Extracting response from agent...")
            
            try:
                ai_response = extract_response_text(response)
                
                if ai_response and ai_response.strip() and "Error extracting response" not in ai_response:
                    logger.info(f"✅ Extracted response: {ai_response[:100]}...")
                    return ai_response.strip()
                else:
                    logger.warning(f"⚠️ No valid response content found: {ai_response}")
                    return "Hi Ignas! I'm here to help with your banking needs. How can I assist you today?"
                
            except Exception as extract_error:
                logger.error(f"❌ Error extracting response: {extract_error}")
                return "I'm sorry, I'm having technical difficulties. Please try again."
            
        except Exception as e:
            logger.error(f"❌ Error getting AI response: {e}")
            return "I'm sorry, I'm having technical difficulties. Please try again."
    
    def send_message_to_ignas(self, text):
        """Send direct message to Ignas."""
        try:
            url = f"{self.api_url}/message/sendText/{self.instance_id}"
            
            payload = {
                "number": self.ignas_jid,  # Direct message to Ignas
                "text": text
            }
            
            logger.info(f"📤 Sending direct message to Ignas")
            logger.info(f"📤 URL: {url}")
            logger.info(f"📤 Payload: {payload}")
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            response_data = response.json()
            logger.info(f"✅ Direct message sent to Ignas. Response: {response_data}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error sending message to Ignas: {e}")
            if hasattr(e, 'response'):
                logger.error(f"❌ Response status: {e.response.status_code}")
                logger.error(f"❌ Response text: {e.response.text}")
            return False
    
    def run(self):
        """Main bot loop."""
        logger.info("🚀 Starting Ignas Direct Bot...")
        
        if not self.setup_ai_agent():
            logger.error("❌ Cannot start without AI agent")
            return
        
        logger.info("🔄 Starting direct message polling loop (every 10s)")
        
        try:
            while True:
                logger.info("🔍 Checking for new messages from Ignas...")
                
                # Get all unprocessed messages from Ignas
                unprocessed_messages = self.get_ignas_messages()
                
                if unprocessed_messages:
                    logger.info(f"📬 Found {len(unprocessed_messages)} unprocessed messages from Ignas")
                    
                    # Process all unprocessed messages (newest first)
                    for message in sorted(unprocessed_messages, key=lambda x: x['timestamp']):
                        logger.info(f"🆕 PROCESSING MESSAGE from {message['name']}: {message['text']}")
                        
                        # Get AI response
                        ai_response = self.get_ai_response(
                            message['text'], 
                            message['name'], 
                            message['phone']
                        )
                        
                        # Send response to Ignas
                        if ai_response:
                            logger.info(f"📤 Sending response to Ignas: {ai_response[:50]}...")
                            success = self.send_message_to_ignas(ai_response)
                            
                            if success:
                                logger.info("✅ Response sent successfully to Ignas")
                            else:
                                logger.error("❌ Failed to send response to Ignas")
                        
                        # Mark this message as processed
                        self.processed_message_ids.add(message['id'])
                        logger.info(f"✅ Marked message {message['id']} as processed")
                        # Update timestamp
                        self.last_processed_timestamp = message['timestamp']
                        
                        # Add delay between processing multiple messages
                        time.sleep(2)
                    
                else:
                    logger.info("📭 No new messages from Ignas")
                
                # Clean up old processed messages to prevent memory issues
                if len(self.processed_message_ids) > 1000:
                    self.processed_message_ids = set(list(self.processed_message_ids)[-500:])
                
                time.sleep(10)  # Wait 10 seconds
                
        except KeyboardInterrupt:
            logger.info("👋 Bot stopped by user")
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}")

def main():
    """Main entry point."""
    print("🤖 Ignas Direct RedBank WhatsApp Bot")
    print("=" * 40)
    
    try:
        bot = IgnasDirectBot()
        bot.run()
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
