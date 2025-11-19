"""
会话管理器：管理用户会话的完整生命周期
整合：DatabaseManager + TokenManager + ShortTermMemoryManager
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
    会话管理器：管理会话的创建、验证、超时和结束

    功能：
    1. 创建新会话（生成token、初始化短期记忆）
    2. 验证token并获取会话
    3. 更新会话活动时间
    4. 30分钟超时检查
    5. 结束会话并触发长期记忆存储
    """

    def __init__(self):
        """初始化会话管理器"""
        self.db = get_db_manager()
        self.token_manager = get_token_manager()

        # 会话短期记忆存储（内存中）
        # key: session_id, value: ShortTermMemoryManager实例
        self.active_memories: Dict[str, ShortTermMemoryManager] = {}

        # 会话对话历史存储（用于结束时的长期存储）
        # key: session_id, value: list of dialogue turns
        self.dialogue_histories: Dict[str, List[Dict[str, str]]] = {}

    def create_session(
        self,
        patient_id: str,
        patient_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        创建新会话

        Args:
            patient_id: 患者ID
            patient_info: 患者信息（从外部医疗系统获取）
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
        # 1. 生成session_id
        session_id = self.token_manager.generate_session_id()

        # 2. 生成URL token
        url_token, expires_at = self.token_manager.generate_url_token(
            session_id=session_id,
            patient_id=patient_id,
            patient_info=patient_info
        )

        # 3. 存储到数据库
        success = self.db.create_session(
            session_id=session_id,
            patient_id=patient_id,
            url_token=url_token,
            token_expires_at=expires_at,
            patient_info=patient_info
        )

        if not success:
            raise RuntimeError(f"创建会话失败: session_id={session_id}")

        # 4. 初始化短期记忆
        self.active_memories[session_id] = ShortTermMemoryManager(
            max_tokens=settings.short_term_max_tokens,
            max_turns=settings.short_term_max_turns
        )

        # 5. 初始化对话历史
        self.dialogue_histories[session_id] = []

        # 6. 生成完整URL
        chat_url = f"http://localhost:{settings.port}/chat/{url_token}"

        print(f"✓ 会话创建成功: {session_id}")
        print(f"  URL: {chat_url}")
        print(f"  过期时间: {expires_at}")

        return {
            "session_id": session_id,
            "url_token": url_token,
            "url": chat_url,
            "expires_at": expires_at
        }

    def get_session_by_token(self, url_token: str) -> Optional[Dict[str, Any]]:
        """
        通过token获取会话信息并验证

        Args:
            url_token: URL中的token

        Returns:
            会话信息字典，如果token无效或过期返回None
        """
        # 1. 验证token
        payload = self.token_manager.verify_token(url_token)
        if not payload:
            return None

        # 2. 从数据库获取会话
        session = self.db.get_session_by_token(url_token)
        if not session:
            return None

        # 3. 检查会话状态
        if session['status'] != 'active':
            print(f"会话已{session['status']}: {session['session_id']}")
            return None

        # 4. 检查是否超过30分钟无活动
        last_activity = datetime.fromisoformat(session['last_activity_at'])
        timeout_threshold = timedelta(minutes=settings.session_timeout_minutes)

        if datetime.utcnow() - last_activity > timeout_threshold:
            print(f"会话超时: {session['session_id']}")
            self.end_session(session['session_id'], reason='timeout')
            return None

        # 5. 更新活动时间
        self.db.update_session_activity(session['session_id'])

        return dict(session)

    def get_short_term_memory(self, session_id: str) -> Optional[ShortTermMemoryManager]:
        """获取会话的短期记忆管理器"""
        return self.active_memories.get(session_id)

    def add_dialogue_turn(
        self,
        session_id: str,
        role: str,
        content: str
    ) -> bool:
        """
        添加对话轮次到短期记忆和对话历史

        Args:
            session_id: 会话ID
            role: "user" 或 "assistant"
            content: 对话内容

        Returns:
            是否触发了短期记忆总结
        """
        # 1. 添加到短期记忆
        memory = self.get_short_term_memory(session_id)
        if not memory:
            print(f"警告: 会话{session_id}的短期记忆不存在，重新创建")
            self.active_memories[session_id] = ShortTermMemoryManager()
            memory = self.active_memories[session_id]

        triggered_summary = memory.add_turn(role, content)

        # 2. 添加到对话历史（用于最终存储）
        if session_id not in self.dialogue_histories:
            self.dialogue_histories[session_id] = []

        self.dialogue_histories[session_id].append({
            "role": role,
            "content": content
        })

        # 3. 更新会话活动时间
        self.db.update_session_activity(session_id)

        return triggered_summary

    def get_dialogue_history(self, session_id: str) -> List[Dict[str, str]]:
        """获取会话的完整对话历史"""
        return self.dialogue_histories.get(session_id, [])

    def end_session(self, session_id: str, reason: str = 'user_request'):
        """
        结束会话

        Args:
            session_id: 会话ID
            reason: 结束原因 ('user_request', 'timeout', 'error')
        """
        print(f"结束会话: {session_id} (原因: {reason})")

        # 1. 更新数据库状态
        self.db.end_session(session_id)

        # 2. 清理内存中的短期记忆
        if session_id in self.active_memories:
            del self.active_memories[session_id]

        # 3. 对话历史保留（用于后续的长期存储流程）
        # 注意：这里不删除 dialogue_histories[session_id]
        # 因为后续需要调用 MemoryStorage 来存储

        print(f"✓ 会话已结束: {session_id}")

    def check_and_cleanup_expired_sessions(self):
        """
        检查并清理所有超时的会话（定时任务）
        """
        expired_sessions = self.db.get_expired_sessions()

        for session in expired_sessions:
            session_id = session['session_id']
            print(f"发现超时会话: {session_id}")
            self.end_session(session_id, reason='timeout')

        return len(expired_sessions)

    def get_session_stats(self, session_id: str) -> Optional[Dict]:
        """
        获取会话统计信息

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


# 全局会话管理器实例
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """获取全局会话管理器实例（单例模式）"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


# 单元测试
if __name__ == "__main__":
    print("=== 会话管理器测试 ===\n")

    sm = SessionManager()

    # 测试1: 创建会话
    print("【测试1 - 创建会话】")
    patient_info = {
        "patient_name": "张三",
        "patient_age": 45,
        "gender": "male",
        "doctor_name": "李医生",
        "department": "心内科",
        "appointment_id": "APT001"
    }

    session_info = sm.create_session(
        patient_id="P12345",
        patient_info=patient_info
    )

    print(f"会话URL: {session_info['url']}")
    session_id = session_info['session_id']
    url_token = session_info['url_token']

    # 测试2: 通过token获取会话
    print("\n【测试2 - 验证Token】")
    session = sm.get_session_by_token(url_token)
    if session:
        print(f"✓ Token验证成功")
        print(f"  患者: {session['patient_name']}")
        print(f"  医生: {session['doctor_name']}")
    else:
        print("✗ Token验证失败")

    # 测试3: 添加对话
    print("\n【测试3 - 添加对话】")
    sm.add_dialogue_turn(session_id, "user", "我最近头痛很厉害")
    sm.add_dialogue_turn(session_id, "assistant", "请问您的头痛是什么时候开始的？")
    sm.add_dialogue_turn(session_id, "user", "大概三天前")

    stats = sm.get_session_stats(session_id)
    print(f"对话轮数: {stats['dialogue_turns']}")
    print(f"短期记忆: {stats['short_term_memory_stats']}")

    # 测试4: 获取对话历史
    print("\n【测试4 - 获取对话历史】")
    history = sm.get_dialogue_history(session_id)
    print(f"对话历史条数: {len(history)}")
    for turn in history:
        print(f"  {turn['role']}: {turn['content'][:30]}...")

    # 测试5: 结束会话
    print("\n【测试5 - 结束会话】")
    sm.end_session(session_id, reason='test_complete')

    # 测试6: 验证会话已结束
    print("\n【测试6 - 验证会话已结束】")
    session = sm.get_session_by_token(url_token)
    print(f"会话状态: {'已结束' if session is None else '仍活跃'}")

    print("\n测试完成")
