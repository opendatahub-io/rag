#!/usr/bin/env python3
"""
Simple RedBank WhatsApp Bot
Reads group messages, processes only the latest non-bot message, and replies with clean AI responses
"""

import os
import time
import json
import requests
import logging
from datetime import datetime
from typing import Optional
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
        logging.FileHandler("simple_redbank_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleRedBankBot:
    """Simple WhatsApp bot for RedBank banking assistance."""
    
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
        
        # Bot state
        self.group_jid = "120363422549041630@g.us"  # RedBank group
        # Track processed message IDs instead of just timestamps
        self.processed_message_ids = set()
        # Start with timestamp from 1 hour ago to catch recent messages
        one_hour_ago = int(datetime.now().timestamp()) - 3600
        self.last_processed_timestamp = one_hour_ago
        logger.info(f"🕐 Bot will process messages newer than: {datetime.fromtimestamp(one_hour_ago)}")
        
        # Initialize LlamaStack
        self.llamastack_url = os.getenv("LLAMASTACK_URL", "http://ragathon-team-1-ragathon-team-1.apps.llama-rag-pool-b84hp.aws.rh-ods.com")
        self.client = None
        self.agent = None
        self.session_id = None
        
        logger.info("🤖 Simple RedBank Bot initialized")
    
    def setup_ai_agent(self):
        """Setup LlamaStack AI agent."""
        try:
            self.client = LlamaStackClient(base_url=self.llamastack_url)
            
            self.agent = Agent(
                self.client,
                model="vllm-inference/llama-4-scout-17b-16e-w4a16",
                instructions="You are a banking assistant; use the MCP tools to fetch user banking information by phone number. Make multiple tool calls to get complete account details including statements and transactions. Only call for info if the user asks for it.",
                tools=[
                    "mcp::redbank-financials", 
                    {
                        "name": "builtin::rag/knowledge_search",
                        "args": {"vector_db_ids": ['vs_1f1dd1b7-49ad-4ceb-8e8d-f0bf9afe2179']},
                    }
                ], 
            )
            
            self.session_id = self.agent.create_session("simple-redbank-bot")
            logger.info("✅ AI agent ready")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error setting up AI agent: {e}")
            return False
    
    def get_unprocessed_user_messages(self):
        """Get all unprocessed user messages from the group."""
        try:
            url = f"{self.api_url}/chat/findMessages/{self.instance_id}"
            
            payload = {
                "where": {"key": {"remoteJid": self.group_jid}}
                # Evolution API pagination doesn't work reliably, get all and sort
            }
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            messages = response.json().get("messages", {}).get("records", [])
            
            logger.info(f"📋 Evolution API returned {len(messages)} messages (sorting to find actual latest)")
            
            if not messages:
                logger.debug("📭 No messages found")
                return None
            
            # Sort messages by timestamp (newest first) 
            sorted_messages = sorted(messages, key=lambda x: x.get("messageTimestamp", 0), reverse=True)
            logger.info(f"🔝 Checking all messages for unprocessed ones")
            
            unprocessed_messages = []
            
            # Check all messages for unprocessed ones
            for message in sorted_messages:
                key = message.get("key", {})
                message_id = key.get("id")
                from_me = key.get("fromMe", False)
                timestamp = message.get("messageTimestamp", 0)
                
                # Skip bot messages
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
                    # Extract phone number
                    participant_pn = key.get("participantPn", "")
                    push_name = key.get("participantPushName", "User")
                    
                    if participant_pn and "@" in participant_pn:
                        phone = participant_pn.split("@")[0]
                        # Format Irish numbers
                        if phone.startswith("353") and len(phone) >= 12:
                            formatted_phone = f"+353 {phone[3:5]} {phone[5:8]} {phone[8:]}"
                        else:
                            formatted_phone = f"+{phone}"
                    else:
                        formatted_phone = "+000000000000"
                    
                    # Skip excluded number
                    if "353834498545" in phone:
                        continue
                    
                    dt = datetime.fromtimestamp(timestamp) if timestamp else "Unknown"
                    logger.info(f"📝 Found unprocessed message: {dt} | text='{text[:30]}...' | id={message_id}")
                    
                    unprocessed_messages.append({
                        "id": message_id,
                        "text": text.strip(),
                        "phone": formatted_phone,
                        "name": push_name,
                        "timestamp": timestamp
                    })
            
            if unprocessed_messages:
                # Return the newest unprocessed message
                newest = max(unprocessed_messages, key=lambda x: x['timestamp'])
                logger.info(f"✅ Returning newest unprocessed message: {newest['id']}")
                return newest
            
            logger.debug("📭 No unprocessed messages found")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting messages: {e}")
            return None
    
    def get_ai_response(self, message_text, sender_name, sender_phone):
        """Get response from AI agent."""
        try:
            user_message = f"Customer '{sender_name}' (phone: {sender_phone}) is asking: {message_text}"
            
            response = self.agent.create_turn(
                messages=[{"role": "user", "content": user_message}],
                session_id=self.session_id
            )
            
            # Extract response using the proven method from test_llamastack_mcp.py
            logger.info("📡 Extracting response from agent...")
            
            try:
                ai_response = extract_response_text(response)
                
                if ai_response and ai_response.strip() and "Error extracting response" not in ai_response:
                    logger.info(f"✅ Extracted response: {ai_response[:100]}...")
                    return ai_response.strip()
                else:
                    logger.warning(f"⚠️ No valid response content found: {ai_response}")
                    return "I'm here to help with your banking needs. How can I assist you today?"
                
            except Exception as extract_error:
                logger.error(f"❌ Error extracting response: {extract_error}")
                return "I'm sorry, I'm having technical difficulties. Please try again."
            
        except Exception as e:
            logger.error(f"❌ Error getting AI response: {e}")
            return "I'm sorry, I'm having technical difficulties. Please try again."
    
    def send_message(self, text):
        """Send message to RedBank group."""
        try:
            url = f"{self.api_url}/message/sendText/{self.instance_id}"
            
            payload = {
                "number": self.group_jid,
                "text": text
            }
            
            logger.info(f"📤 Sending to URL: {url}")
            logger.info(f"📤 Payload: {payload}")
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            response_data = response.json()
            logger.info(f"✅ Message sent successfully. Response: {response_data}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error sending message: {e}")
            logger.error(f"❌ Response status: {getattr(e.response, 'status_code', 'N/A')}")
            logger.error(f"❌ Response text: {getattr(e.response, 'text', 'N/A')}")
            return False
    
    def run(self):
        """Main bot loop."""
        logger.info("🚀 Starting Simple RedBank Bot...")
        
        if not self.setup_ai_agent():
            logger.error("❌ Cannot start without AI agent")
            return
        
        logger.info("🔄 Starting polling loop (every 10s)")
        
        try:
            while True:
                logger.info("🔍 Checking for new messages...")
                
                # Get latest unprocessed user message
                latest_message = self.get_unprocessed_user_messages()
                
                if latest_message:
                    logger.info(f"🆕 NEW MESSAGE from {latest_message['name']} ({latest_message['phone']}): {latest_message['text']}")
                    
                    # Get AI response
                    ai_response = self.get_ai_response(
                        latest_message['text'], 
                        latest_message['name'], 
                        latest_message['phone']
                    )
                    
                    # Send response
                    if ai_response:
                        logger.info(f"📤 Sending response: {ai_response[:50]}...")
                        self.send_message(ai_response)
                        
                        # Mark this message as processed
                        self.processed_message_ids.add(latest_message['id'])
                        logger.info(f"✅ Added message {latest_message['id']} to processed set (now {len(self.processed_message_ids)} processed)")
                        # Update timestamp to this message's timestamp
                        self.last_processed_timestamp = latest_message['timestamp']
                    
                else:
                    logger.info("📭 No new messages")
                
                # Clean up old processed messages to prevent memory issues
                if len(self.processed_message_ids) > 1000:
                    # Keep only the last 500 processed message IDs
                    self.processed_message_ids = set(list(self.processed_message_ids)[-500:])
                
                time.sleep(10)  # Wait 10 seconds
                
        except KeyboardInterrupt:
            logger.info("👋 Bot stopped by user")
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}")

def main():
    """Main entry point."""
    print("🤖 Simple RedBank WhatsApp Bot")
    print("=" * 40)
    
    try:
        bot = SimpleRedBankBot()
        bot.run()
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
