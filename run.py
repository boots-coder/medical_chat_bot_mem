#!/usr/bin/env python3
"""
Medical Chat Memory Manager 启动脚本
启动 FastAPI 应用
"""
import uvicorn
from backend.core.config import settings

if __name__ == "__main__":
    print("="*60)
    print("医疗咨询系统启动中...")
    print(f"访问地址: http://{settings.host}:{settings.port}")
    print(f"API文档: http://{settings.host}:{settings.port}/docs")
    print("="*60)

    uvicorn.run(
        "backend.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )
