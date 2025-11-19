#!/usr/bin/env python3
"""
Medical Chat Memory Manager Startup Script
Start FastAPI application
"""
import uvicorn
from backend.core.config import settings

if __name__ == "__main__":
    print("="*60)
    print("Medical Consultation System starting...")
    print(f"Access URL: http://{settings.host}:{settings.port}")
    print(f"API Docs: http://{settings.host}:{settings.port}/docs")
    print("="*60)

    uvicorn.run(
        "backend.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )
