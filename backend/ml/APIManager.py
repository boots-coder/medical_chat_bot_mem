import json
from openai import OpenAI
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass


@dataclass
class APIConfig:
    """API configuration dataclass"""
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "openai/gpt-4o-mini"
    default_headers: Dict[str, str] = None

    def __post_init__(self):
        if self.default_headers is None:
            # For ARK API, no additional headers needed, use empty dict
            self.default_headers = {}


class APIManager:
    """
    Unified API Manager: Manages all LLM API calls

    Features:
    1. Unified configuration management (API keys, models, base_url, etc.)
    2. Standardized API call interface
    3. Error handling and retry mechanisms
    4. Supports different call modes (JSON, text, streaming, etc.)
    5. Easy to switch models or API providers
    """

    def __init__(self, config: APIConfig = None):
        """
        Initialize API Manager

        Args:
            config: API configuration, uses default if None
        """
        if config is None:
            # Use new default configuration (Doubao/ByteDance ARK API)
            config = APIConfig(
                api_key="137bf427-389e-4e02-a0b7-d0c1c01b2787",
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                model="deepseek-v3-250324"
            )

        self.config = config
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize OpenAI client"""
        if "YOUR_OPENROUTER_API_KEY" in self.config.api_key:
            print("⚠️ Warning: API key not set")
            self.client = None
        else:
            try:
                self.client = OpenAI(
                    base_url=self.config.base_url,
                    api_key=self.config.api_key,
                    default_headers=self.config.default_headers
                )
                print(f"✅ API client initialized successfully: {self.config.model}")
            except Exception as e:
                print(f"❌ API client initialization failed: {e}")
                self.client = None

    def is_available(self) -> bool:
        """Check if API is available"""
        return self.client is not None

    def call_completion(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict] = None,
        temperature: float = 0.1,
        max_tokens: int = 500,
        model_override: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Standard completion API call

        Args:
            messages: Conversation message list
            response_format: Response format (e.g., {"type": "json_object"})
            temperature: Temperature parameter
            max_tokens: Maximum token count
            model_override: Temporarily override model name

        Returns:
            {
                "success": bool,
                "content": str,
                "usage": dict,
                "error": str
            }
        """
        if not self.client:
            return {
                "success": False,
                "content": "",
                "usage": {},
                "error": "API client not initialized"
            }

        try:
            # Build API call parameters
            call_params = {
                "model": model_override or self.config.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            # Add response format if specified
            if response_format:
                call_params["response_format"] = response_format

            # Call API
            completion = self.client.chat.completions.create(**call_params)

            # Extract results
            content = completion.choices[0].message.content
            usage = completion.usage.model_dump() if completion.usage else {}

            # Debug output removed, can be re-enabled if needed
            # print(f"🔍 API response content: {repr(content)}")

            return {
                "success": True,
                "content": content,
                "usage": usage,
                "error": ""
            }

        except Exception as e:
            error_msg = f"API call failed: {e}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "content": "",
                "usage": {},
                "error": error_msg
            }

    def call_json_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 300
    ) -> Optional[Dict]:
        """
        API call specifically for JSON responses (simplified interface)

        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Temperature parameter
            max_tokens: Maximum token count

        Returns:
            Parsed JSON object, returns None on failure
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        result = self.call_completion(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens
        )

        if not result["success"]:
            return None

        try:
            content = result["content"]

            # Handle ARK API response wrapped in ```json```
            if content.startswith("```json"):
                # Extract content between ```json``` and ```
                start = content.find("```json") + 7  # Skip ```json
                end = content.rfind("```")
                if end > start:
                    content = content[start:end].strip()
            elif content.startswith("```"):
                # Handle other code block formats
                start = content.find("\n") + 1
                end = content.rfind("```")
                if end > start:
                    content = content[start:end].strip()

            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"🔍 Raw content: {result['content']}")
            return None

    def call_text_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 800
    ) -> Optional[str]:
        """
        API call specifically for text responses (simplified interface)

        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Temperature parameter
            max_tokens: Maximum token count

        Returns:
            Generated text content, returns None on failure
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        result = self.call_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return result["content"] if result["success"] else None

    def update_config(self, **kwargs):
        """
        Update API configuration

        Args:
            **kwargs: Configuration items to update (api_key, model, base_url, etc.)
        """
        updated = False

        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                updated = True
                print(f"✅ Configuration updated: {key} = {value}")

        if updated:
            self._initialize_client()

    def switch_to_ark_api(self):
        """Quick switch to ByteDance ARK API"""
        self.update_config(
            api_key="137bf427-389e-4e02-a0b7-d0c1c01b2787",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            model="deepseek-v3-250324"
        )
        print("✅ Switched to ByteDance ARK API (deepseek-v3-250324)")

    def switch_to_openrouter_api(self):
        """Quick switch to OpenRouter API"""
        self.update_config(
            api_key="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",
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-4o-mini"
        )
        print("✅ Switched to OpenRouter API (gpt-4o-mini)")

    def get_config_info(self) -> Dict:
        """Get current configuration information"""
        return {
            "model": self.config.model,
            "base_url": self.config.base_url,
            "api_key_masked": f"{self.config.api_key[:10]}...{self.config.api_key[-10:]}",
            "client_available": self.is_available()
        }


# Global singleton instance (optional use)
_global_api_manager = None

def get_api_manager() -> APIManager:
    """Get global API manager instance (singleton pattern)"""
    global _global_api_manager
    if _global_api_manager is None:
        _global_api_manager = APIManager()
    return _global_api_manager


# Unit tests
def test_api_manager():
    """API Manager unit tests"""
    print("="*60)
    print("API Manager Unit Tests")
    print("="*60)

    # Test initialization
    api_manager = APIManager()
    print(f"\n✅ Initialization complete")
    print(f"Configuration info: {api_manager.get_config_info()}")

    if not api_manager.is_available():
        print("❌ API not available, skipping functional tests")
        return

    # Test JSON call
    print(f"\n【Test JSON Call】")
    system_prompt = """You are a test assistant. Please return JSON format: {"test": true, "message": "Test successful"}"""
    user_prompt = "Please return JSON as requested"

    json_result = api_manager.call_json_completion(system_prompt, user_prompt)
    if json_result:
        print(f"✅ JSON call successful: {json_result}")
    else:
        print(f"❌ JSON call failed")

    # Test text call
    print(f"\n【Test Text Call】")
    system_prompt = "You are a concise assistant, reply in no more than 20 words"
    user_prompt = "What is an API?"

    text_result = api_manager.call_text_completion(system_prompt, user_prompt)
    if text_result:
        print(f"✅ Text call successful: {text_result}")
    else:
        print(f"❌ Text call failed")

    # Test API switching
    print(f"\n【Test API Switching】")
    print("Current configuration:", api_manager.get_config_info())

    # Can test switching to OpenRouter (if needed)
    # api_manager.switch_to_openrouter_api()
    # print("Configuration after switch:", api_manager.get_config_info())

    # Test global singleton
    print(f"\n【Test Global Singleton】")
    global_manager = get_api_manager()
    print(f"Global instance available: {global_manager.is_available()}")
    print(f"Is same instance: {global_manager is get_api_manager()}")

    print("\n" + "="*60)
    print("Tests complete")
    print("="*60)


if __name__ == "__main__":
    test_api_manager()
