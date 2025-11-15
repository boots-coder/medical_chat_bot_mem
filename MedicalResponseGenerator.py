import json
from typing import Optional, Dict, Any
from APIManager import APIManager
from ShortTermMemoryManager import ShortTermMemoryManager
from RAGIntentClassifier import RAGIntentClassifier
from TestData import get_test_scenario


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
        return """你是一个专业的医疗AI助手，具备短期记忆和长期记忆能力。

## 你的能力
- 短期记忆：记住当前对话的所有内容和历史摘要
- 长期记忆：能够检索用户的历史就诊记录、症状变化、用药历史
- 个性化建议：基于用户的完整病史提供针对性建议

## 回复原则
1. **安全第一**：严重症状建议立即就医，不提供确诊
2. **个性化**：充分利用历史信息，提供个性化建议  
3. **连贯性**：回复要与对话上下文保持连贯
4. **专业性**：使用准确的医学术语，但确保患者理解

## 信息整合策略
- 如果有长期记忆信息，要重点对比当前症状与历史记录
- 如果用户询问历史用药，要结合历史记录给出建议
- 如果是新症状，主要基于当前描述和一般医学知识

## 回复格式
请按以下结构回复：

**症状分析：** [简要分析当前症状]

**个性化建议：** [基于历史记录的针对性建议]

**用药指导：** [具体的用药建议]

**注意事项：** [需要注意的事项]

注意：如果没有长期记忆信息，就基于短期记忆和当前查询正常回复。"""
    
    def generate_response(
        self, 
        user_query: str, 
        long_term_memory: str = "", 
        mock_mode: bool = False
    ) -> Dict[str, Any]:
        """
        生成医疗回复
        
        Args:
            user_query: 用户当前查询
            long_term_memory: 长期记忆上下文（如果需要RAG）
            mock_mode: 是否为测试模式（使用模拟的RAG结果）
            
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
        
        # 1. 获取短期记忆上下文
        short_memory_context = self.memory_manager.get_context()
        
        # 2. RAG意图分类
        rag_result = self.rag_classifier.classify_rag_intent(user_query, short_memory_context)
        rag_triggered = rag_result["need_rag"] if rag_result else False
        confidence = rag_result["confidence"] if rag_result else 0.0
        
        # 3. 确定长期记忆内容
        final_long_memory = ""
        if rag_triggered:
            if mock_mode and long_term_memory:
                # 测试模式：使用提供的模拟长期记忆
                final_long_memory = long_term_memory
            elif not mock_mode:
                # 生产模式：这里应该调用实际的RAG检索
                final_long_memory = "【生产模式】这里应该调用实际的RAG检索系统"
        
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


# 完整集成测试
def test_complete_system():
    """完整系统集成测试：使用TestData中的场景"""
    print("="*70)
    print("完整系统集成测试")
    print("="*70)
    
    generator = MedicalResponseGenerator()
    
    # 测试场景1：需要RAG的对话
    print("\n【测试场景1：需要RAG的对话】")
    print("-" * 50)
    
    rag_scenario = get_test_scenario("rag_required")
    generator.new_session()  # 开始新session
    
    # 模拟完整对话过程，但不包括最后一轮
    conversation = rag_scenario["conversation"][:-1]  # 除了最后一轮
    for turn in conversation:
        generator.add_conversation_turn(turn["role"], turn["content"])
        print(f"{turn['role'].upper()}: {turn['content']}")
    
    # 最后一轮是需要RAG的查询
    final_query = rag_scenario["final_query"]
    mock_rag_data = rag_scenario["mock_rag_result"]
    
    print(f"\n👤 用户最终查询：{final_query}")
    print("\n🤖 系统处理中...")
    
    # 生成回复（使用模拟的RAG数据）
    result = generator.generate_response(
        user_query=final_query,
        long_term_memory=mock_rag_data,
        mock_mode=True
    )
    
    print(f"\n✅ RAG触发：{result['rag_triggered']}")
    print(f"✅ 置信度：{result['confidence']:.2f}")
    print(f"\n🩺 AI回复：\n{result['response']}")
    
    # 测试场景2：不需要RAG的对话
    print("\n" + "="*70)
    print("\n【测试场景2：不需要RAG的对话】")
    print("-" * 50)
    
    no_rag_scenario = get_test_scenario("no_rag")
    generator.new_session()  # 开始新session
    
    # 模拟完整对话过程，但不包括最后一轮
    conversation = no_rag_scenario["conversation"][:-1]  # 除了最后一轮
    for turn in conversation:
        generator.add_conversation_turn(turn["role"], turn["content"])
        print(f"{turn['role'].upper()}: {turn['content']}")
    
    # 最后一轮是不需要RAG的查询
    final_query = no_rag_scenario["final_query"]
    
    print(f"\n👤 用户最终查询：{final_query}")
    print("\n🤖 系统处理中...")
    
    # 生成回复（不应该触发RAG）
    result = generator.generate_response(
        user_query=final_query,
        mock_mode=True
    )
    
    print(f"\n✅ RAG触发：{result['rag_triggered']}")
    print(f"✅ 置信度：{result['confidence']:.2f}")
    print(f"\n🩺 AI回复：\n{result['response']}")
    
    # 显示最终记忆状态
    print(f"\n【系统记忆状态】")
    print(generator.get_memory_stats())
    
    print("\n" + "="*70)
    print("集成测试完成")
    print("="*70)


if __name__ == "__main__":
    test_complete_system()