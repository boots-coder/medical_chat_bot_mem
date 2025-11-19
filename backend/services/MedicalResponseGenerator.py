import json
from typing import Optional, Dict, Any
from backend.ml.APIManager import APIManager
from backend.models.ShortTermMemoryManager import ShortTermMemoryManager
from backend.ml.RAGIntentClassifier import RAGIntentClassifier


class MedicalResponseGenerator:
    """
    医疗回复生成器：整合三个信息源生成最终回复
    
    信息源：
    1. 用户当前查询
    2. 短期记忆（来自ShortTermMemoryManager）
    3. 长期记忆（如果RAGIntentClassifier判断需要RAG）
    
    集成所有组件，生成个性化的医疗建议
    """
    
    def __init__(self, api_manager: APIManager = None):
        """
        初始化医疗回复生成器
        
        Args:
            api_manager: API管理器实例，如果为None则创建默认实例
        """
        # 初始化API管理器
        self.api_manager = api_manager if api_manager else APIManager()
        
        # 初始化各组件，共享同一个API管理器
        self.memory_manager = ShortTermMemoryManager(max_tokens=2000, max_turns=8, api_manager=self.api_manager)
        self.rag_classifier = RAGIntentClassifier(api_manager=self.api_manager)
        
        if not self.api_manager.is_available():
            print("⚠️ 警告: API不可用，医疗回复生成功能将无法使用")
    
    def _build_system_prompt(self) -> str:
        """构建医疗回复生成的系统提示词"""
        return """你是专业的医疗AI助手，具备记忆能力。

## 回复原则
1. **简洁精准**：直接回答问题，避免冗余信息
2. **安全第一**：严重症状建议就医，不提供确诊
3. **个性化**：利用历史信息提供针对性建议
4. **通俗易懂**：避免过多医学术语，患者能理解即可

## 回复要求
- 每次回复控制在 3-5 句话内
- 直接回答用户问题，不要过度展开
- 如有历史记录，简要对比即可
- 必要时才给出用药建议，不要主动推荐药物
- 避免使用"症状分析"、"个性化建议"等格式化标题

## 示例风格
用户："我头痛怎么办？"
✅ 好的回复："根据您的描述，可能是紧张性头痛。建议先休息，保持安静环境。如果疼痛加剧或持续超过24小时，建议就医。"
❌ 差的回复："**症状分析：**您出现了头痛症状，这可能是由多种原因引起的...**个性化建议：**...**用药指导：**..."

保持回复自然、简洁、有针对性。"""
    
    def generate_response(
        self,
        user_query: str,
        short_term_context: str = "",
        long_term_memory: str = ""
    ) -> Dict[str, Any]:
        """
        生成医疗回复

        Args:
            user_query: 用户当前查询
            short_term_context: 短期记忆上下文（从SessionManager获取）
            long_term_memory: 长期记忆上下文（由MemoryRetrieval检索）

        Returns:
            {
                "response": str,           # 生成的回复
                "used_short_memory": str,  # 使用的短期记忆
                "used_long_memory": str,   # 使用的长期记忆
                "rag_triggered": bool,     # 是否触发了RAG
                "confidence": float        # RAG分类置信度
            }
        """
        if not self.api_manager.is_available():
            return {
                "response": "错误：API不可用",
                "used_short_memory": "",
                "used_long_memory": "",
                "rag_triggered": False,
                "confidence": 0.0
            }

        # 1. 使用传入的短期记忆上下文
        short_memory_context = short_term_context
        
        # 2. RAG意图分类
        rag_result = self.rag_classifier.classify_rag_intent(user_query, short_memory_context)
        rag_triggered = rag_result["need_rag"] if rag_result else False
        confidence = rag_result["confidence"] if rag_result else 0.0
        
        # 3. 使用传入的长期记忆内容（已经由外部RAG检索完成）
        final_long_memory = long_term_memory if long_term_memory else ""
        
        # 4. 构建完整的上下文提示
        context_parts = []
        
        if short_memory_context:
            context_parts.append(f"短期记忆上下文：\n{short_memory_context}")
        
        if final_long_memory:
            context_parts.append(f"长期记忆上下文：\n{final_long_memory}")
        
        context_parts.append(f"用户当前查询：{user_query}")
        
        full_context = "\n\n".join(context_parts)
        
        # 5. 调用LLM生成回复
        response = self.api_manager.call_text_completion(
            system_prompt=self._build_system_prompt(),
            user_prompt=full_context,
            temperature=0.3,  # 稍高一些，保持回复的自然性
            max_tokens=800
        )
        
        if response:
            return {
                "response": response,
                "used_short_memory": short_memory_context,
                "used_long_memory": final_long_memory,
                "rag_triggered": rag_triggered,
                "confidence": confidence
            }
        else:
            return {
                "response": "抱歉，系统暂时无法生成回复",
                "used_short_memory": short_memory_context,
                "used_long_memory": final_long_memory,
                "rag_triggered": rag_triggered,
                "confidence": confidence
            }
    
    def add_conversation_turn(self, role: str, content: str):
        """添加对话轮次到短期记忆"""
        self.memory_manager.add_turn(role, content)
    
    def new_session(self):
        """开始新的session（清空短期记忆）"""
        self.memory_manager.clear()
        print("[系统] 新session开始，短期记忆已清空")
    
    def get_memory_stats(self) -> Dict:
        """获取记忆状态统计"""
        return self.memory_manager.get_stats()
