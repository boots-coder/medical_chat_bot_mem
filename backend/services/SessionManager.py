"""
Session Manager: Manages the complete lifecycle of user sessions
Integration: DatabaseManager + TokenManager + ShortTermMemoryManager
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import asyncio

from backend.core.DatabaseManager import get_db_manager
from backend.services.TokenManager import get_token_manager
from backend.models.ShortTermMemoryManager import ShortTermMemoryManager
from backend.core.config import settings


class SessionManager:
    """
    Session Manager: Manages session creation, validation, timeout and termination

    Features:
    1. Create new session (generate token, initialize short-term memory)
    2. Validate token and retrieve session
    3. Update session activity time
    4. 30-minute timeout check
    5. End session and trigger long-term memory storage
    """

    def __init__(self):
        """Initialize session manager"""
        self.db = get_db_manager()
        self.token_manager = get_token_manager()

        # Session short-term memory storage (in memory)
        # key: session_id, value: ShortTermMemoryManager instance
        self.active_memories: Dict[str, ShortTermMemoryManager] = {}

        # Session dialogue history storage (for long-term storage at session end)
        # key: session_id, value: list of dialogue turns
        self.dialogue_histories: Dict[str, List[Dict[str, str]]] = {}

    def create_session(
        self,
        patient_id: str,
        patient_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create new session

        Args:
            patient_id: Patient ID
            patient_info: Patient information (obtained from external medical system)
                {
                    "patient_name": str,
                    "patient_age": int,
                    "gender": str,
                    "doctor_name": str,
                    "department": str,
                    "appointment_id": str
                }

        Returns:
            {
                "session_id": str,
                "url_token": str,
                "url": str,
                "expires_at": datetime
            }
        """
        # 1. Generate session_id
        session_id = self.token_manager.generate_session_id()

        # 2. Generate URL token
        url_token, expires_at = self.token_manager.generate_url_token(
            session_id=session_id,
            patient_id=patient_id,
            patient_info=patient_info
        )

        # 3. Store to database
        success = self.db.create_session(
            session_id=session_id,
            patient_id=patient_id,
            url_token=url_token,
            token_expires_at=expires_at,
            patient_info=patient_info
        )

        if not success:
            raise RuntimeError(f"Failed to create session: session_id={session_id}")

        # 4. Initialize short-term memory
        self.active_memories[session_id] = ShortTermMemoryManager(
            max_tokens=settings.short_term_max_tokens,
            max_turns=settings.short_term_max_turns
        )

        # 5. Initialize dialogue history
        self.dialogue_histories[session_id] = []

        # 6. Generate complete URL
        chat_url = f"http://localhost:{settings.port}/chat/{url_token}"

        print(f"✓ Session created successfully: {session_id}")
        print(f"  URL: {chat_url}")
        print(f"  Expires at: {expires_at}")

        return {
            "session_id": session_id,
            "url_token": url_token,
            "url": chat_url,
            "expires_at": expires_at
        }

    def get_session_by_token(self, url_token: str) -> Optional[Dict[str, Any]]:
        """
        Get session information by token and validate

        Args:
            url_token: Token in URL

        Returns:
            Session information dictionary, returns None if token is invalid or expired
        """
        # 1. Validate token
        payload = self.token_manager.verify_token(url_token)
        if not payload:
            return None

        # 2. Get session from database
        session = self.db.get_session_by_token(url_token)
        if not session:
            return None

        # 3. Check session status
        if session['status'] != 'active':
            print(f"Session is {session['status']}: {session['session_id']}")
            return None

        # 4. Check if inactive for more than 30 minutes
        last_activity = datetime.fromisoformat(session['last_activity_at'])
        timeout_threshold = timedelta(minutes=settings.session_timeout_minutes)

        if datetime.utcnow() - last_activity > timeout_threshold:
            print(f"Session timeout: {session['session_id']}")
            self.end_session(session['session_id'], reason='timeout')
            return None

        # 5. Update activity time
        self.db.update_session_activity(session['session_id'])

        return dict(session)

    def get_short_term_memory(self, session_id: str) -> Optional[ShortTermMemoryManager]:
        """Get the short-term memory manager for the session"""
        return self.active_memories.get(session_id)

    def add_dialogue_turn(
        self,
        session_id: str,
        role: str,
        content: str
    ) -> bool:
        """
        Add dialogue turn to short-term memory and dialogue history

        Args:
            session_id: Session ID
            role: "user" or "assistant"
            content: Dialogue content

        Returns:
            Whether short-term memory summarization was triggered
        """
        # 1. Add to short-term memory
        memory = self.get_short_term_memory(session_id)
        if not memory:
            print(f"Warning: Short-term memory for session {session_id} does not exist, recreating")
            self.active_memories[session_id] = ShortTermMemoryManager()
            memory = self.active_memories[session_id]

        triggered_summary = memory.add_turn(role, content)

        # 2. Add to dialogue history (for final storage)
        if session_id not in self.dialogue_histories:
            self.dialogue_histories[session_id] = []

        self.dialogue_histories[session_id].append({
            "role": role,
            "content": content
        })

        # 3. Update session activity time
        self.db.update_session_activity(session_id)

        return triggered_summary

    def get_dialogue_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get the complete dialogue history for the session"""
        return self.dialogue_histories.get(session_id, [])

    def end_session(self, session_id: str, reason: str = 'user_request'):
        """
        End session

        Args:
            session_id: Session ID
            reason: Reason for ending ('user_request', 'timeout', 'error')
        """
        print(f"Ending session: {session_id} (reason: {reason})")

        # 1. Update database status
        self.db.end_session(session_id)

        # 2. Clean up short-term memory in memory
        if session_id in self.active_memories:
            del self.active_memories[session_id]

        # 3. Preserve dialogue history (for subsequent long-term storage process)
        # Note: Do not delete dialogue_histories[session_id] here
        # because it is needed later to call MemoryStorage for storage

        print(f"✓ Session ended: {session_id}")

    def check_and_cleanup_expired_sessions(self):
        """
        Check and clean up all expired sessions (scheduled task)
        """
        expired_sessions = self.db.get_expired_sessions()

        for session in expired_sessions:
            session_id = session['session_id']
            print(f"Found expired session: {session_id}")
            self.end_session(session_id, reason='timeout')

        return len(expired_sessions)

    def get_session_stats(self, session_id: str) -> Optional[Dict]:
        """
        Get session statistics

        Returns:
            {
                "dialogue_turns": int,
                "short_term_memory_stats": dict,
                "is_active": bool
            }
        """
        memory = self.get_short_term_memory(session_id)
        if not memory:
            return None

        return {
            "dialogue_turns": len(self.dialogue_histories.get(session_id, [])),
            "short_term_memory_stats": memory.get_stats(),
            "is_active": session_id in self.active_memories
        }


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get global session manager instance (singleton pattern)"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


# Unit tests
if __name__ == "__main__":
    print("=== Session Manager Tests ===\n")

    sm = SessionManager()

    # Test 1: Create session
    print("【Test 1 - Create Session】")
    patient_info = {
        "patient_name": "Zhang San",
        "patient_age": 45,
        "gender": "male",
        "doctor_name": "Dr. Li",
        "department": "Cardiology",
        "appointment_id": "APT001"
    }

    session_info = sm.create_session(
        patient_id="P12345",
        patient_info=patient_info
    )

    print(f"Session URL: {session_info['url']}")
    session_id = session_info['session_id']
    url_token = session_info['url_token']

    # Test 2: Get session by token
    print("\n【Test 2 - Validate Token】")
    session = sm.get_session_by_token(url_token)
    if session:
        print(f"✓ Token validation successful")
        print(f"  Patient: {session['patient_name']}")
        print(f"  Doctor: {session['doctor_name']}")
    else:
        print("✗ Token validation failed")

    # Test 3: Add dialogue
    print("\n【Test 3 - Add Dialogue】")
    sm.add_dialogue_turn(session_id, "user", "I've been having severe headaches lately")
    sm.add_dialogue_turn(session_id, "assistant", "When did your headaches start?")
    sm.add_dialogue_turn(session_id, "user", "About three days ago")

    stats = sm.get_session_stats(session_id)
    print(f"Dialogue turns: {stats['dialogue_turns']}")
    print(f"Short-term memory: {stats['short_term_memory_stats']}")

    # Test 4: Get dialogue history
    print("\n【Test 4 - Get Dialogue History】")
    history = sm.get_dialogue_history(session_id)
    print(f"Dialogue history count: {len(history)}")
    for turn in history:
        print(f"  {turn['role']}: {turn['content'][:30]}...")

    # Test 5: End session
    print("\n【Test 5 - End Session】")
    sm.end_session(session_id, reason='test_complete')

    # Test 6: Verify session ended
    print("\n【Test 6 - Verify Session Ended】")
    session = sm.get_session_by_token(url_token)
    print(f"Session status: {'Ended' if session is None else 'Still active'}")

    print("\n Tests completed")
