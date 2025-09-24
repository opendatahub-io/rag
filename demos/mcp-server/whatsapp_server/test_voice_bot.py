#!/usr/bin/env python3
"""
Test script for the RedBank bot voice message functionality
"""

import os
import sys
from dotenv import load_dotenv

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Load environment variables
load_dotenv("whatsapp_server/.env")

from redbank_bot import RedBankBot

def test_voice_detection():
    """Test voice message detection with sample data."""
    
    bot = RedBankBot()
    
    # Sample message with voice/audio content
    sample_voice_message = {
        "audioMessage": {
            "url": "https://example.com/voice.ogg",
            "mimetype": "audio/ogg; codecs=opus",
            "fileLength": 12345,
            "seconds": 5,
            "ptt": True,  # Push-to-talk (voice note)
            "mediaKey": "sample_key",
            "fileEncSha256": "sample_hash"
        }
    }
    
    sample_text_message = {
        "conversation": "Hello, this is a text message"
    }
    
    # Test voice message detection
    print("🧪 Testing voice message detection...")
    print(f"Voice message detected: {bot.is_voice_message(sample_voice_message)}")
    print(f"Text message detected: {bot.is_voice_message(sample_text_message)}")
    
    # Test message text extraction
    print("\n🧪 Testing message text extraction...")
    voice_text = bot.extract_message_text(sample_voice_message)
    text_text = bot.extract_message_text(sample_text_message)
    print(f"Voice message text: {voice_text}")
    print(f"Text message text: {text_text}")
    
    # Test voice message info extraction
    print("\n🧪 Testing voice message info extraction...")
    voice_info = bot.get_voice_message_info(sample_voice_message)
    print(f"Voice info: {voice_info}")
    
    print("\n✅ Voice detection tests completed!")

if __name__ == "__main__":
    print("🎤 RedBank Bot Voice Message Test")
    print("=" * 40)
    test_voice_detection()
