"""
API server for the Market Research SMOL Agents system
"""

import uvicorn
import logging
from pathlib import Path
import os

from .routes import app
from ..config.settings import API_HOST, API_PORT, API_WORKERS

logger = logging.getLogger("api.server")

def run_server():
    """Run the API server"""
    logger.info(f"Starting API server on {API_HOST}:{API_PORT}")
    
    # Ensure necessary directories exist
    from ..config.settings import DATA_DIR, REPORT_DIR
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # Start the server
    uvicorn.run(
        "api.routes:app",
        host=API_HOST,
        port=API_PORT,
        workers=API_WORKERS,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()