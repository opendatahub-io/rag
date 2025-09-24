#!/usr/bin/env python3
"""
Test voice message reply functionality
"""

import requests
import base64
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv("whatsapp_server/.env")

def test_voice_reply():
    """Test sending a simple voice message."""
    
    api_url = os.getenv("EVOLUTION_API_URL", "https://api.evoapicloud.com")
    instance_id = os.getenv("EVOLUTION_API_ID")
    api_token = os.getenv("EVOLUTION_API_TOKEN")
    group_jid = "120363422549041630@g.us"  # RedBank group
    
    headers = {
        "apikey": api_token,
        "Content-Type": "application/json"
    }
    
    # Create a simple audio message (we'll use a placeholder base64)
    # In a real scenario, you'd have actual audio data
    print("🎤 Testing voice message send capability...")
    
    # Test with a simple audio URL instead of base64
    url = f"{api_url}/message/sendWhatsAppAudio/{instance_id}"
    
    # Method 1: Try with a public audio URL
    payload = {
        "number": group_jid,
        "audio": "https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Voice message sent successfully!")
        else:
            print("❌ Failed to send voice message")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_voice_reply()
