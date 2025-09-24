#!/usr/bin/env python3
"""
Startup script for the WhatsApp RedBank Bot server.
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main entry point for the WhatsApp server."""
    
    # Check if .env file exists
    env_file = project_root / "whatsapp_server" / ".env"
    env_example = project_root / "whatsapp_server" / "env.example"
    
    if not env_file.exists() and env_example.exists():
        print("⚠️  .env file not found!")
        print(f"📝 Please copy {env_example} to {env_file} and configure your Evolution API credentials.")
        print("\nExample:")
        print(f"  cp {env_example} {env_file}")
        print(f"  # Then edit {env_file} with your actual credentials")
        return 1
    
    # Set environment variables for the app
    os.environ.setdefault("PYTHONPATH", str(project_root))
    
    print("🚀 Starting WhatsApp RedBank Bot server...")
    print("📱 The server will listen for WhatsApp messages from RedBank group")
    print("💬 Auto-reply: 'Welcome to RedBank'")
    print("🔗 Evolution API integration enabled")
    print("\n" + "="*50)
    
    # Start the server
    try:
        uvicorn.run(
            "whatsapp_server.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
