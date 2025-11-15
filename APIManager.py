import json
from openai import OpenAI
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass


@dataclass
class APIConfig:
    """API配置数据类"""
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "openai/gpt-4o-mini"
    default_headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.default_headers is None:
            # 对于ARK API，不需要额外的headers，使用空字典
            self.default_headers = {}


class APIManager:
    """
    统一API管理类：管理所有LLM API调用
    
    功能：
    1. 统一配置管理（API密钥、模型、base_url等）
    2. 标准化的API调用接口
    3. 错误处理和重试机制
    4. 支持不同的调用模式（JSON、文本、流式等）
    5. 便于后续切换模型或API提供商
    """
    
    def __init__(self, config: APIConfig = None):
        """
        初始化API管理器
        
        Args:
            config: API配置，如果为None则使用默认配置
        """
        if config is None:
            # 使用新的默认配置（豆包/字节跳动 ARK API）
            config = APIConfig(
                api_key="137bf427-389e-4e02-a0b7-d0c1c01b2787",
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                model="deepseek-v3-250324"
            )
        
        self.config = config
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """初始化OpenAI客户端"""
        if "YOUR_OPENROUTER_API_KEY" in self.config.api_key:
            print("⚠️ 警告: API密钥未设置")
            self.client = None
        else:
            try:
                self.client = OpenAI(
                    base_url=self.config.base_url,
                    api_key=self.config.api_key,
                    default_headers=self.config.default_headers
                )
                print(f"✅ API客户端初始化成功：{self.config.model}")
            except Exception as e:
                print(f"❌ API客户端初始化失败：{e}")
                self.client = None
    
    def is_available(self) -> bool:
        """检查API是否可用"""
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
        标准的completion API调用
        
        Args:
            messages: 对话消息列表
            response_format: 响应格式（如{"type": "json_object"}）
            temperature: 温度参数
            max_tokens: 最大token数
            model_override: 临时覆盖模型名称
            
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
                "error": "API客户端未初始化"
            }
        
        try:
            # 构建API调用参数
            call_params = {
                "model": model_override or self.config.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # 如果指定了响应格式，添加到参数中
            if response_format:
                call_params["response_format"] = response_format
            
            # 调用API
            completion = self.client.chat.completions.create(**call_params)
            
            # 提取结果
            content = completion.choices[0].message.content
            usage = completion.usage.model_dump() if completion.usage else {}
            
            # 调试输出已移除，如需调试可重新启用
            # print(f"🔍 API返回内容: {repr(content)}")
            
            return {
                "success": True,
                "content": content,
                "usage": usage,
                "error": ""
            }
            
        except Exception as e:
            error_msg = f"API调用失败：{e}"
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
        专门用于JSON响应的API调用（简化接口）
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            解析后的JSON对象，失败返回None
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
            
            # 处理ARK API返回的被```json```包裹的JSON
            if content.startswith("```json"):
                # 提取```json```和```之间的内容
                start = content.find("```json") + 7  # 跳过```json
                end = content.rfind("```")
                if end > start:
                    content = content[start:end].strip()
            elif content.startswith("```"):
                # 处理其他代码块格式
                start = content.find("\n") + 1
                end = content.rfind("```")
                if end > start:
                    content = content[start:end].strip()
            
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败：{e}")
            print(f"🔍 原始内容：{result['content']}")
            return None
    
    def call_text_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 800
    ) -> Optional[str]:
        """
        专门用于文本响应的API调用（简化接口）
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            生成的文本内容，失败返回None
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
        更新API配置
        
        Args:
            **kwargs: 要更新的配置项（api_key, model, base_url等）
        """
        updated = False
        
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                updated = True
                print(f"✅ 更新配置：{key} = {value}")
        
        if updated:
            self._initialize_client()
    
    def switch_to_ark_api(self):
        """快速切换到字节跳动ARK API"""
        self.update_config(
            api_key="137bf427-389e-4e02-a0b7-d0c1c01b2787",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            model="deepseek-v3-250324"
        )
        print("✅ 已切换到字节跳动ARK API (deepseek-v3-250324)")
    
    def switch_to_openrouter_api(self):
        """快速切换到OpenRouter API"""
        self.update_config(
            api_key="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",
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-4o-mini"
        )
        print("✅ 已切换到OpenRouter API (gpt-4o-mini)")
    
    def get_config_info(self) -> Dict:
        """获取当前配置信息"""
        return {
            "model": self.config.model,
            "base_url": self.config.base_url,
            "api_key_masked": f"{self.config.api_key[:10]}...{self.config.api_key[-10:]}",
            "client_available": self.is_available()
        }


# 全局单例实例（可选使用）
_global_api_manager = None

def get_api_manager() -> APIManager:
    """获取全局API管理器实例（单例模式）"""
    global _global_api_manager
    if _global_api_manager is None:
        _global_api_manager = APIManager()
    return _global_api_manager


# 单元测试
def test_api_manager():
    """API管理器单元测试"""
    print("="*60)
    print("API管理器单元测试")
    print("="*60)
    
    # 测试初始化
    api_manager = APIManager()
    print(f"\n✅ 初始化完成")
    print(f"配置信息：{api_manager.get_config_info()}")
    
    if not api_manager.is_available():
        print("❌ API不可用，跳过功能测试")
        return
    
    # 测试JSON调用
    print(f"\n【测试JSON调用】")
    system_prompt = """你是一个测试助手。请返回JSON格式：{"test": true, "message": "测试成功"}"""
    user_prompt = "请按要求返回JSON"
    
    json_result = api_manager.call_json_completion(system_prompt, user_prompt)
    if json_result:
        print(f"✅ JSON调用成功：{json_result}")
    else:
        print(f"❌ JSON调用失败")
    
    # 测试文本调用
    print(f"\n【测试文本调用】")
    system_prompt = "你是一个简洁的助手，回复不超过20个字"
    user_prompt = "什么是API？"
    
    text_result = api_manager.call_text_completion(system_prompt, user_prompt)
    if text_result:
        print(f"✅ 文本调用成功：{text_result}")
    else:
        print(f"❌ 文本调用失败")
    
    # 测试API切换
    print(f"\n【测试API切换】")
    print("当前配置：", api_manager.get_config_info())
    
    # 可以测试切换到OpenRouter（如果需要）
    # api_manager.switch_to_openrouter_api()
    # print("切换后配置：", api_manager.get_config_info())
    
    # 测试全局单例
    print(f"\n【测试全局单例】")
    global_manager = get_api_manager()
    print(f"全局实例可用：{global_manager.is_available()}")
    print(f"是否为同一实例：{global_manager is get_api_manager()}")
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)


if __name__ == "__main__":
    test_api_manager()