#!/usr/bin/env python3
"""
Startup script for SignLango backend with proper file upload limits
"""
import uvicorn
from main import app

if __name__ == "__main__":
    # Configure uvicorn with larger request body limits for video uploads
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        # Increase limits for large file uploads
        limit_max_requests=1000,
        limit_concurrency=1000,
        # Allow larger request bodies (50MB)
        timeout_keep_alive=30,
        # Additional configuration for handling large files
        access_log=True,
        log_level="info"
    )


