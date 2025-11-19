"""
Token Manager: Generate and verify JWT tokens
Used to create secure one-time session URLs
"""
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from backend.core.config import settings


class TokenManager:
    """Token Manager: Generate and verify JWT tokens"""

    def __init__(self):
        """Initialize Token Manager"""
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
        Generate URL token (JWT format)

        Args:
            session_id: Session ID
            patient_id: Patient ID
            patient_info: Patient information (from external medical system)

        Returns:
            (token string, expiration time)
        """
        # Calculate expiration time
        expires_at = datetime.utcnow() + timedelta(minutes=self.expire_minutes)

        # Build JWT payload
        payload = {
            "session_id": session_id,
            "patient_id": patient_id,
            "patient_info": patient_info,
            "exp": expires_at,
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(16)  # JWT ID, prevents replay attacks
        }

        # Generate JWT token
        token = jwt.encode(
            payload,
            self.secret_key,
            algorithm=self.algorithm
        )

        return token, expires_at

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify token and return payload

        Args:
            token: JWT token string

        Returns:
            Decoded payload, returns None if verification fails
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload

        except jwt.ExpiredSignatureError:
            print("Token has expired")
            return None

        except jwt.InvalidTokenError as e:
            print(f"Invalid token: {e}")
            return None

    def generate_session_id(self) -> str:
        """
        Generate unique session ID

        Returns:
            session_id string
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        random_part = secrets.token_urlsafe(8)
        return f"S_{timestamp}_{random_part}"

    def is_token_expired(self, token: str) -> bool:
        """
        Check if token is expired (without verifying signature)

        Args:
            token: JWT token string

        Returns:
            True if expired, False if not expired
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


# Global Token Manager instance
_token_manager: Optional[TokenManager] = None


def get_token_manager() -> TokenManager:
    """Get global Token Manager instance (singleton pattern)"""
    global _token_manager
    if _token_manager is None:
        _token_manager = TokenManager()
    return _token_manager


# Unit tests
if __name__ == "__main__":
    print("=== Token Manager Test ===\n")

    tm = TokenManager()

    # Test 1: Generate token
    print("【Test 1 - Generate Token】")
    session_id = tm.generate_session_id()
    print(f"Generated session_id: {session_id}")

    patient_info = {
        "patient_name": "John Doe",
        "patient_age": 45,
        "gender": "male",
        "doctor_name": "Dr. Smith",
        "department": "Cardiology"
    }

    token, expires_at = tm.generate_url_token(
        session_id=session_id,
        patient_id="P12345",
        patient_info=patient_info
    )

    print(f"Generated Token: {token[:50]}...")
    print(f"Expiration time: {expires_at}")

    # Test 2: Verify token
    print("\n【Test 2 - Verify Token】")
    payload = tm.verify_token(token)
    if payload:
        print("✓ Token verification successful")
        print(f"  session_id: {payload['session_id']}")
        print(f"  patient_id: {payload['patient_id']}")
        print(f"  patient_name: {payload['patient_info']['patient_name']}")
    else:
        print("✗ Token verification failed")

    # Test 3: Verify invalid token
    print("\n【Test 3 - Verify Invalid Token】")
    invalid_token = "invalid.token.here"
    payload = tm.verify_token(invalid_token)
    print(f"Invalid token verification result: {'✗ Correctly rejected' if payload is None else '✓ Incorrectly accepted'}")

    # Test 4: Check expiration
    print("\n【Test 4 - Check Token Expiration】")
    is_expired = tm.is_token_expired(token)
    print(f"Is token expired: {'Yes' if is_expired else 'No'}")

    print("\nTest complete")
