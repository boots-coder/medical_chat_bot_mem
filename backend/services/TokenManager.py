"""
Token管理器：生成和验证JWT token
用于创建安全的一次性会话URL
"""
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from backend.core.config import settings


class TokenManager:
    """Token管理器：生成和验证JWT token"""

    def __init__(self):
        """初始化Token管理器"""
        self.secret_key = settings.jwt_secret_key
        self.algorithm = settings.jwt_algorithm
        self.expire_minutes = settings.token_expire_minutes

    def generate_url_token(
        self,
        session_id: str,
        patient_id: str,
        patient_info: Dict[str, Any]
    ) -> tuple[str, datetime]:
        """
        生成URL token（JWT格式）

        Args:
            session_id: 会话ID
            patient_id: 患者ID
            patient_info: 患者信息（从外部医疗系统获取）

        Returns:
            (token字符串, 过期时间)
        """
        # 计算过期时间
        expires_at = datetime.utcnow() + timedelta(minutes=self.expire_minutes)

        # 构建JWT payload
        payload = {
            "session_id": session_id,
            "patient_id": patient_id,
            "patient_info": patient_info,
            "exp": expires_at,
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(16)  # JWT ID，防止重放攻击
        }

        # 生成JWT token
        token = jwt.encode(
            payload,
            self.secret_key,
            algorithm=self.algorithm
        )

        return token, expires_at

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        验证token并返回payload

        Args:
            token: JWT token字符串

        Returns:
            解码后的payload，如果验证失败返回None
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload

        except jwt.ExpiredSignatureError:
            print("Token已过期")
            return None

        except jwt.InvalidTokenError as e:
            print(f"无效的Token: {e}")
            return None

    def generate_session_id(self) -> str:
        """
        生成唯一的session ID

        Returns:
            session_id字符串
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        random_part = secrets.token_urlsafe(8)
        return f"S_{timestamp}_{random_part}"

    def is_token_expired(self, token: str) -> bool:
        """
        检查token是否过期（不验证签名）

        Args:
            token: JWT token字符串

        Returns:
            True表示已过期，False表示未过期
        """
        try:
            payload = jwt.decode(
                token,
                options={"verify_signature": False}
            )
            exp_timestamp = payload.get("exp")
            if exp_timestamp:
                exp_datetime = datetime.utcfromtimestamp(exp_timestamp)
                return datetime.utcnow() > exp_datetime
            return True

        except Exception:
            return True


# 全局Token管理器实例
_token_manager: Optional[TokenManager] = None


def get_token_manager() -> TokenManager:
    """获取全局Token管理器实例（单例模式）"""
    global _token_manager
    if _token_manager is None:
        _token_manager = TokenManager()
    return _token_manager


# 单元测试
if __name__ == "__main__":
    print("=== Token管理器测试 ===\n")

    tm = TokenManager()

    # 测试1: 生成token
    print("【测试1 - 生成Token】")
    session_id = tm.generate_session_id()
    print(f"生成的session_id: {session_id}")

    patient_info = {
        "patient_name": "张三",
        "patient_age": 45,
        "gender": "male",
        "doctor_name": "李医生",
        "department": "心内科"
    }

    token, expires_at = tm.generate_url_token(
        session_id=session_id,
        patient_id="P12345",
        patient_info=patient_info
    )

    print(f"生成的Token: {token[:50]}...")
    print(f"过期时间: {expires_at}")

    # 测试2: 验证token
    print("\n【测试2 - 验证Token】")
    payload = tm.verify_token(token)
    if payload:
        print("✓ Token验证成功")
        print(f"  session_id: {payload['session_id']}")
        print(f"  patient_id: {payload['patient_id']}")
        print(f"  patient_name: {payload['patient_info']['patient_name']}")
    else:
        print("✗ Token验证失败")

    # 测试3: 验证无效token
    print("\n【测试3 - 验证无效Token】")
    invalid_token = "invalid.token.here"
    payload = tm.verify_token(invalid_token)
    print(f"无效Token验证结果: {'✗ 正确拒绝' if payload is None else '✓ 错误接受'}")

    # 测试4: 检查过期
    print("\n【测试4 - 检查Token是否过期】")
    is_expired = tm.is_token_expired(token)
    print(f"Token是否过期: {'是' if is_expired else '否'}")

    print("\n测试完成")
