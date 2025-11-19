"""
FastAPI主应用：医疗咨询系统后端
提供REST API和WebSocket接口
"""
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
import asyncio
import json

from backend.core.config import settings
from backend.services.SessionManager import get_session_manager
from backend.services.MemoryStorage import get_memory_storage
from backend.services.MemoryRetrieval import get_memory_retrieval
from backend.services.MedicalResponseGenerator import MedicalResponseGenerator


# ==================== Pydantic Models ====================

class CreateSessionRequest(BaseModel):
    """创建会话请求"""
    patient_id: str
    patient_name: str
    patient_age: int
    gender: str
    doctor_name: str
    department: str
    appointment_id: str


class CreateSessionResponse(BaseModel):
    """创建会话响应"""
    session_id: str
    url: str
    url_token: str
    expires_at: datetime


class ChatMessage(BaseModel):
    """聊天消息"""
    message: str


class ChatResponse(BaseModel):
    """聊天响应"""
    response: str
    session_id: str
    used_short_memory: bool
    used_long_memory: bool
    rag_triggered: bool
    confidence: float


class SessionSummaryResponse(BaseModel):
    """会话摘要响应"""
    session_id: str
    patient_id: str
    dialogue_turns: int
    start_time: str
    end_time: str
    status: str


# ==================== FastAPI Application ====================

app = FastAPI(
    title="医疗咨询系统API",
    description="智能医疗对话系统，支持短期记忆、长期记忆和RAG检索",
    version="1.0.0",
    debug=settings.debug
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 获取项目根目录
BASE_DIR = Path(__file__).parent.parent.parent

# 静态文件服务
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "frontend" / "static")), name="static")

# Jinja2 模板
templates = Jinja2Templates(directory=str(BASE_DIR / "frontend" / "templates"))

# 初始化管理器
session_manager = get_session_manager()
memory_storage = get_memory_storage()
memory_retrieval = get_memory_retrieval()
response_generator = MedicalResponseGenerator()


# ==================== 模拟外部医疗系统API ====================

@app.post("/api/external/create-session", response_model=CreateSessionResponse)
async def external_create_session(request: CreateSessionRequest):
    """
    外部医疗系统调用此接口创建会话

    流程：
    1. 生成session_id和URL token
    2. 存储到数据库
    3. 返回URL给外部系统（外部系统通过短信/邮件发送给患者）
    """
    try:
        patient_info = {
            "patient_name": request.patient_name,
            "patient_age": request.patient_age,
            "gender": request.gender,
            "doctor_name": request.doctor_name,
            "department": request.department,
            "appointment_id": request.appointment_id
        }

        session_info = session_manager.create_session(
            patient_id=request.patient_id,
            patient_info=patient_info
        )

        return CreateSessionResponse(
            session_id=session_info['session_id'],
            url=session_info['url'],
            url_token=session_info['url_token'],
            expires_at=session_info['expires_at']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建会话失败: {str(e)}")


# ==================== 聊天界面入口 ====================

@app.get("/chat/{url_token}")
async def chat_page(request: Request, url_token: str):
    """
    患者点击URL后进入的聊天页面

    验证token → 返回聊天界面HTML
    """
    # 验证token
    session = session_manager.get_session_by_token(url_token)

    if not session:
        return HTMLResponse(
            content="<h1>链接已失效</h1><p>此会话已过期或不存在，请联系医生获取新的链接。</p>",
            status_code=403
        )

    # 使用Jinja2模板渲染聊天界面
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "session_id": session['session_id'],
        "patient_name": session['patient_name'],
        "doctor_name": session['doctor_name'],
        "department": session['department'],
        "ws_port": settings.port
    })


# ==================== WebSocket实时对话 ====================

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket连接：实时对话

    流程：
    1. 接收用户消息
    2. 添加到短期记忆
    3. RAG检索（如果需要）
    4. 生成回复
    5. 发送回复给用户
    """
    await websocket.accept()
    print(f"WebSocket连接建立: {session_id}")

    try:
        while True:
            # 接收消息
            data = await websocket.receive_text()
            message_data = json.loads(data) if isinstance(data, str) else data

            if message_data.get('type') == 'user':
                user_message = message_data['content']

                # 1. 添加到短期记忆
                session_manager.add_dialogue_turn(session_id, "user", user_message)

                # 2. 获取短期记忆上下文
                memory = session_manager.get_short_term_memory(session_id)
                short_term_context = memory.get_context() if memory else ""

                # 3. 获取会话信息
                # TODO: 需要从session_manager获取patient_id
                # 临时方案：从数据库查询
                from backend.core.DatabaseManager import get_db_manager
                db = get_db_manager()
                cursor = db.sqlite_conn.cursor()
                cursor.execute("SELECT patient_id FROM sessions WHERE session_id = ?", (session_id,))
                row = cursor.fetchone()
                patient_id = row[0] if row else None

                # 4. 生成回复
                response_text = await generate_response_async(
                    patient_id=patient_id,
                    user_query=user_message,
                    short_term_context=short_term_context,
                    session_id=session_id
                )

                # 5. 添加到短期记忆
                session_manager.add_dialogue_turn(session_id, "assistant", response_text)

                # 6. 发送回复
                await websocket.send_json({
                    "type": "assistant",
                    "content": response_text
                })

    except WebSocketDisconnect:
        print(f"WebSocket连接断开: {session_id}")


async def generate_response_async(
    patient_id: str,
    user_query: str,
    short_term_context: str,
    session_id: str
) -> str:
    """
    异步生成回复

    整合：短期记忆 + 长期记忆检索 + LLM回复生成
    """
    # 1. 长期记忆检索
    long_term_memory = ""
    if patient_id:
        retrieval_result = memory_retrieval.retrieve(
            patient_id=patient_id,
            user_query=user_query,
            short_term_context=short_term_context,
            n_results=3
        )

        if retrieval_result['rag_triggered']:
            long_term_memory = retrieval_result['formatted_context']

    # 2. 生成回复
    result = response_generator.generate_response(
        user_query=user_query,
        short_term_context=short_term_context,
        long_term_memory=long_term_memory
    )

    return result['response']


# ==================== 会话管理API ====================

@app.post("/api/session/{session_id}/end")
async def end_session(session_id: str):
    """
    结束会话

    流程：
    1. 获取对话历史
    2. 触发长期记忆存储
    3. 更新会话状态
    """
    try:
        # 1. 获取对话历史
        dialogue_history = session_manager.get_dialogue_history(session_id)

        if not dialogue_history:
            raise HTTPException(status_code=404, detail="会话不存在或无对话记录")

        # 2. 获取会话信息
        from backend.core.DatabaseManager import get_db_manager
        db = get_db_manager()
        cursor = db.sqlite_conn.cursor()
        cursor.execute("SELECT patient_id, created_at FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="会话不存在")

        patient_id = row[0]
        start_time = row[1]
        end_time = datetime.utcnow().isoformat()

        # 3. 存储长期记忆（异步执行，不阻塞）
        asyncio.create_task(store_memory_async(
            session_id=session_id,
            patient_id=patient_id,
            dialogue_list=dialogue_history,
            start_time=start_time,
            end_time=end_time
        ))

        # 4. 结束会话
        session_manager.end_session(session_id, reason='user_request')

        return {"status": "success", "message": "会话已结束"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"结束会话失败: {str(e)}")


async def store_memory_async(
    session_id: str,
    patient_id: str,
    dialogue_list: List[Dict],
    start_time: str,
    end_time: str
):
    """异步存储长期记忆"""
    try:
        memory_storage.store_session_memory(
            session_id=session_id,
            patient_id=patient_id,
            dialogue_list=dialogue_list,
            start_time=start_time,
            end_time=end_time
        )
        print(f"✓ 长期记忆存储完成: {session_id}")
    except Exception as e:
        print(f"✗ 长期记忆存储失败: {e}")


@app.get("/api/session/{session_id}/summary")
async def get_session_summary(session_id: str):
    """
    获取会话摘要（供外部医疗系统调用）

    返回：对话轮数、开始/结束时间、状态
    """
    from backend.core.DatabaseManager import get_db_manager
    db = get_db_manager()
    cursor = db.sqlite_conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="会话不存在")

    session = dict(row)

    return {
        "session_id": session['session_id'],
        "patient_id": session['patient_id'],
        "patient_name": session['patient_name'],
        "doctor_name": session['doctor_name'],
        "department": session['department'],
        "status": session['status'],
        "created_at": session['created_at'],
        "ended_at": session.get('ended_at'),
        "last_activity_at": session['last_activity_at']
    }


# ==================== 健康检查 ====================

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "医疗咨询系统API",
        "docs": "/docs",
        "health": "/health",
        "test": "/test"
    }


@app.get("/test")
async def test_page(request: Request):
    """外部医疗系统测试页面"""
    return templates.TemplateResponse("external_test.html", {"request": request})


# ==================== 启动函数 ====================

if __name__ == "__main__":
    import uvicorn

    print("="*60)
    print("医疗咨询系统启动中...")
    print(f"访问地址: http://{settings.host}:{settings.port}")
    print(f"API文档: http://{settings.host}:{settings.port}/docs")
    print("="*60)

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )
