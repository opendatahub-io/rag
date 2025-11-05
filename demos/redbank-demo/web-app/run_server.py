#!/usr/bin/env python3
"""
Entry point script for the Voice Processing API.
Run this script from the web-app directory to start the FastAPI server.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )
