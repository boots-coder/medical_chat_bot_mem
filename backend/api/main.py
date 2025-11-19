"""
FastAPI Main Application: Medical Consultation System Backend
Provides REST API and WebSocket interfaces
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
    """Create session request"""
    patient_id: str
    patient_name: str
    patient_age: int
    gender: str
    doctor_name: str
    department: str
    appointment_id: str


class CreateSessionResponse(BaseModel):
    """Create session response"""
    session_id: str
    url: str
    url_token: str
    expires_at: datetime


class ChatMessage(BaseModel):
    """Chat message"""
    message: str


class ChatResponse(BaseModel):
    """Chat response"""
    response: str
    session_id: str
    used_short_memory: bool
    used_long_memory: bool
    rag_triggered: bool
    confidence: float


class SessionSummaryResponse(BaseModel):
    """Session summary response"""
    session_id: str
    patient_id: str
    dialogue_turns: int
    start_time: str
    end_time: str
    status: str


# ==================== FastAPI Application ====================

app = FastAPI(
    title="Medical Consultation System API",
    description="Intelligent medical dialogue system with short-term memory, long-term memory, and RAG retrieval",
    version="1.0.0",
    debug=settings.debug
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get project root directory
BASE_DIR = Path(__file__).parent.parent.parent

# Static file service
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "frontend" / "static")), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory=str(BASE_DIR / "frontend" / "templates"))

# Initialize managers
session_manager = get_session_manager()
memory_storage = get_memory_storage()
memory_retrieval = get_memory_retrieval()
response_generator = MedicalResponseGenerator()


# ==================== Simulated External Medical System API ====================

@app.post("/api/external/create-session", response_model=CreateSessionResponse)
async def external_create_session(request: CreateSessionRequest):
    """
    External medical system calls this interface to create a session

    Process:
    1. Generate session_id and URL token
    2. Store in database
    3. Return URL to external system (external system sends to patient via SMS/email)
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
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


# ==================== Chat Interface Entry Point ====================

@app.get("/chat/{url_token}")
async def chat_page(request: Request, url_token: str):
    """
    Chat page patients enter after clicking URL

    Verify token → Return chat interface HTML
    """
    # Verify token
    session = session_manager.get_session_by_token(url_token)

    if not session:
        return HTMLResponse(
            content="<h1>Link Expired</h1><p>This session has expired or does not exist. Please contact your doctor for a new link.</p>",
            status_code=403
        )

    # Render chat interface using Jinja2 template
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "session_id": session['session_id'],
        "patient_name": session['patient_name'],
        "doctor_name": session['doctor_name'],
        "department": session['department'],
        "ws_port": settings.port
    })


# ==================== WebSocket Real-time Dialogue ====================

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket connection: Real-time dialogue

    Process:
    1. Receive user message
    2. Add to short-term memory
    3. RAG retrieval (if needed)
    4. Generate response
    5. Send response to user
    """
    await websocket.accept()
    print(f"WebSocket connection established: {session_id}")

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data) if isinstance(data, str) else data

            if message_data.get('type') == 'user':
                user_message = message_data['content']

                # 1. Add to short-term memory
                session_manager.add_dialogue_turn(session_id, "user", user_message)

                # 2. Get short-term memory context
                memory = session_manager.get_short_term_memory(session_id)
                short_term_context = memory.get_context() if memory else ""

                # 3. Get session information
                # TODO: Need to get patient_id from session_manager
                # Temporary solution: query from database
                from backend.core.DatabaseManager import get_db_manager
                db = get_db_manager()
                cursor = db.sqlite_conn.cursor()
                cursor.execute("SELECT patient_id FROM sessions WHERE session_id = ?", (session_id,))
                row = cursor.fetchone()
                patient_id = row[0] if row else None

                # 4. Generate response
                response_text = await generate_response_async(
                    patient_id=patient_id,
                    user_query=user_message,
                    short_term_context=short_term_context,
                    session_id=session_id
                )

                # 5. Add to short-term memory
                session_manager.add_dialogue_turn(session_id, "assistant", response_text)

                # 6. Send response
                await websocket.send_json({
                    "type": "assistant",
                    "content": response_text
                })

    except WebSocketDisconnect:
        print(f"WebSocket connection disconnected: {session_id}")


async def generate_response_async(
    patient_id: str,
    user_query: str,
    short_term_context: str,
    session_id: str
) -> str:
    """
    Asynchronously generate response

    Integrate: short-term memory + long-term memory retrieval + LLM response generation
    """
    # 1. Long-term memory retrieval
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

    # 2. Generate response
    result = response_generator.generate_response(
        user_query=user_query,
        short_term_context=short_term_context,
        long_term_memory=long_term_memory
    )

    return result['response']


# ==================== Session Management API ====================

@app.post("/api/session/{session_id}/end")
async def end_session(session_id: str):
    """
    End session

    Process:
    1. Get dialogue history
    2. Trigger long-term memory storage
    3. Update session status
    """
    try:
        # 1. Get dialogue history
        dialogue_history = session_manager.get_dialogue_history(session_id)

        if not dialogue_history:
            raise HTTPException(status_code=404, detail="Session does not exist or has no dialogue records")

        # 2. Get session information
        from backend.core.DatabaseManager import get_db_manager
        db = get_db_manager()
        cursor = db.sqlite_conn.cursor()
        cursor.execute("SELECT patient_id, created_at FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Session does not exist")

        patient_id = row[0]
        start_time = row[1]
        end_time = datetime.utcnow().isoformat()

        # 3. Store long-term memory (asynchronously, non-blocking)
        asyncio.create_task(store_memory_async(
            session_id=session_id,
            patient_id=patient_id,
            dialogue_list=dialogue_history,
            start_time=start_time,
            end_time=end_time
        ))

        # 4. End session
        session_manager.end_session(session_id, reason='user_request')

        return {"status": "success", "message": "Session ended"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to end session: {str(e)}")


async def store_memory_async(
    session_id: str,
    patient_id: str,
    dialogue_list: List[Dict],
    start_time: str,
    end_time: str
):
    """Asynchronously store long-term memory"""
    try:
        memory_storage.store_session_memory(
            session_id=session_id,
            patient_id=patient_id,
            dialogue_list=dialogue_list,
            start_time=start_time,
            end_time=end_time
        )
        print(f"✓ Long-term memory storage complete: {session_id}")
    except Exception as e:
        print(f"✗ Long-term memory storage failed: {e}")


@app.get("/api/session/{session_id}/summary")
async def get_session_summary(session_id: str):
    """
    Get session summary (called by external medical system)

    Returns: dialogue turns, start/end time, status
    """
    from backend.core.DatabaseManager import get_db_manager
    db = get_db_manager()
    cursor = db.sqlite_conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Session does not exist")

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


@app.get("/api/session/{session_id}/memory-summary")
async def get_session_memory_summary(session_id: str):
    """
    Get session's long-term memory summary

    Read complete dialogue analysis results from vector database
    """
    from backend.core.DatabaseManager import get_db_manager

    db = get_db_manager()

    # Query long-term memory for this session from Chroma
    results = db.chroma_collection.get(
        where={"session_id": session_id},
        include=["metadatas", "documents"]
    )

    if not results['ids']:
        raise HTTPException(status_code=404, detail="Long-term memory not found for this session")

    # Parse first record (typically one record per session)
    metadata = results['metadatas'][0]
    document = results['documents'][0]
    analysis = json.loads(metadata['analysis_json'])

    return {
        "session_id": session_id,
        "patient_id": metadata['patient_id'],
        "session_topic": analysis.get('session_topic', ''),
        "narrative_summary": analysis.get('narrative_summary', ''),
        "main_complaint": analysis.get('main_complaint_vectorized', ''),
        "dialogue_rounds": analysis.get('dialogue_rounds', 0),
        "start_time": metadata['created_at'],
        "end_time": metadata['end_time'],
        "knowledge_graph": {
            "entities_count": len(analysis.get('knowledge_graph', {}).get('entities', [])),
            "relationships_count": len(analysis.get('knowledge_graph', {}).get('relationships', []))
        }
    }


# ==================== Health Check ====================

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    """Root path"""
    return {
        "message": "Medical Consultation System API",
        "docs": "/docs",
        "health": "/health",
        "test": "/test"
    }


@app.get("/test")
async def test_page(request: Request):
    """External medical system test page"""
    return templates.TemplateResponse("external_test.html", {"request": request})


# ==================== Startup Function ====================

if __name__ == "__main__":
    import uvicorn

    print("="*60)
    print("Medical Consultation System starting...")
    print(f"Access URL: http://{settings.host}:{settings.port}")
    print(f"API Docs: http://{settings.host}:{settings.port}/docs")
    print("="*60)

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )
