import json
from typing import List, Dict, Optional
from APIManager import APIManager


class ShortTermMemoryManager:
    """
    短期记忆管理器：实现类似LangChain的buffer功能
    
    功能：
    1. 维护当前session的对话窗口
    2. 当token超过限制时进行总结
    3. 保留历史摘要 + 当前窗口
    4. 为RAG意图分类和最终回复提供短期记忆上下文
    """
    
    def __init__(self, max_tokens: int = 2000, max_turns: int = 10, api_manager: APIManager = None):
        """
        Args:
            max_tokens: 最大token数，超过则触发总结
            max_turns: 最大对话轮数，超过则触发总结
            api_manager: API管理器实例，如果为None则创建默认实例
        """
        # 记忆管理参数
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        
        # 记忆状态
        self.historical_summary = ""  # 历史摘要
        self.current_window = []      # 当前窗口对话
        self.estimated_tokens = 0     # 估算的token数
        
        # 初始化API管理器
        self.api_manager = api_manager if api_manager else APIManager()
        if not self.api_manager.is_available():
            print("⚠️ 警告: API不可用，短期记忆总结功能将无法使用")
    
    def _estimate_tokens(self, text: str) -> int:
        """
        粗略估算文本的token数（中文字符 * 1.5）
        TODO: 未来可用tiktoken精确计算
        """
        return int(len(text) * 1.5)
    
    def _build_summarization_prompt(self) -> str:
        """构建总结提示词"""
        return """你是一个医疗对话总结助手，需要将多轮医患对话总结为简洁的上下文摘要。

## 总结原则
1. 保留所有医疗相关的关键信息：症状、病史、用药、诊断建议
2. 保持时间顺序和因果关系
3. 删除无关的寒暄和重复信息
4. 总结长度控制在200字以内

## 输出格式
返回JSON格式：
```json
{
  "summary": "简洁的对话摘要，包含关键医疗信息",
  "key_points": ["要点1", "要点2", "要点3"]
}
```

要求：
- summary: 连贯的叙述性总结
- key_points: 3-5个关键要点列表"""
    
    def _summarize_conversations(self) -> Optional[str]:
        """
        总结当前窗口的对话
        
        Returns:
            总结后的摘要文本，失败返回None
        """
        if not self.api_manager.is_available() or not self.current_window:
            return None
        
        # 构建对话文本
        dialogue_text = ""
        for i, turn in enumerate(self.current_window, 1):
            role = "用户" if turn["role"] == "user" else "助手"
            dialogue_text += f"[轮次{i} - {role}]: {turn['content']}\n\n"
        
        # 如果有历史摘要，也包含进去
        context_text = ""
        if self.historical_summary:
            context_text = f"历史摘要：{self.historical_summary}\n\n当前对话：\n{dialogue_text}"
        else:
            context_text = f"对话内容：\n{dialogue_text}"
        
        # 使用API管理器进行调用
        result = self.api_manager.call_json_completion(
            system_prompt=self._build_summarization_prompt(),
            user_prompt=context_text,
            temperature=0.1,
            max_tokens=300
        )
        
        if result:
            return result.get("summary", "")
        else:
            print("对话总结失败")
            return None
    
    def add_turn(self, role: str, content: str) -> bool:
        """
        添加新的对话轮次
        
        Args:
            role: "user" 或 "assistant"
            content: 对话内容
            
        Returns:
            是否触发了总结
        """
        # 添加到当前窗口
        turn = {"role": role, "content": content}
        self.current_window.append(turn)
        
        # 更新token估计
        self.estimated_tokens += self._estimate_tokens(content)
        
        # 检查是否需要总结
        triggered_summary = False
        if (self.estimated_tokens > self.max_tokens or 
            len(self.current_window) >= self.max_turns * 2):  # *2因为一轮=用户+助手
            
            # 进行总结
            new_summary = self._summarize_conversations()
            if new_summary:
                self.historical_summary = new_summary
                self.current_window = []  # 清空当前窗口
                self.estimated_tokens = self._estimate_tokens(self.historical_summary)
                triggered_summary = True
                print(f"[短期记忆] 触发总结，新摘要长度: {len(self.historical_summary)}字符")
        
        return triggered_summary
    
    def get_context(self) -> str:
        """
        获取完整的短期记忆上下文
        
        Returns:
            包含历史摘要 + 当前窗口的完整上下文
        """
        context_parts = []
        
        # 添加历史摘要
        if self.historical_summary:
            context_parts.append(f"历史摘要：{self.historical_summary}")
        
        # 添加当前窗口
        if self.current_window:
            current_text = ""
            for turn in self.current_window:
                role = "用户" if turn["role"] == "user" else "助手"
                current_text += f"{role}：{turn['content']}\n"
            context_parts.append(f"当前对话：\n{current_text.strip()}")
        
        return "\n\n".join(context_parts) if context_parts else ""
    
    def get_stats(self) -> Dict:
        """获取记忆状态统计"""
        return {
            "历史摘要长度": len(self.historical_summary),
            "当前窗口轮数": len(self.current_window),
            "估算token数": self.estimated_tokens,
            "是否有历史摘要": bool(self.historical_summary)
        }
    
    def clear(self):
        """清空所有记忆（新session开始时调用）"""
        self.historical_summary = ""
        self.current_window = []
        self.estimated_tokens = 0


# 单元测试
def test_short_term_memory():
    """短期记忆管理器单元测试"""
    print("="*60)
    print("短期记忆管理器单元测试")
    print("="*60)
    
    # 创建实例，设置较小的限制便于测试
    memory = ShortTermMemoryManager(max_tokens=500, max_turns=3)
    
    print("\n【测试场景】模拟一个逐渐增长的医疗对话")
    print("-" * 40)
    
    # 模拟对话序列
    conversation = [
        ("user", "我最近头痛很厉害，特别是太阳穴的位置"),
        ("assistant", "请问您的头痛是什么时候开始的？是持续性还是阵发性的？"),
        ("user", "大概三天前开始，一阵一阵的疼，特别是下午的时候"),
        ("assistant", "听起来可能是偏头痛。您之前有过类似的情况吗？有没有恶心呕吐的症状？"),
        ("user", "以前也有过，但没这么严重。有点恶心，但没有呕吐"),
        ("assistant", "建议您可以服用布洛芬缓解症状，同时注意休息，避免强光刺激"),
        ("user", "布洛芬要怎么吃？一天几次？"),
        ("assistant", "一般是每次400mg，一天2-3次，餐后服用。如果症状持续不缓解，建议就医"),
        ("user", "好的，谢谢医生。对了，我还想问问，是不是和我最近工作压力大有关系？"),
        ("assistant", "确实有可能。压力、睡眠不足、饮食不规律都可能诱发偏头痛。建议您调整作息")
    ]
    
    # 逐步添加对话
    for i, (role, content) in enumerate(conversation, 1):
        print(f"\n[轮次 {i}] {role}: {content[:50]}...")
        
        # 添加对话
        triggered = memory.add_turn(role, content)
        
        # 显示状态
        stats = memory.get_stats()
        print(f"   状态: 窗口={stats['当前窗口轮数']}, tokens≈{stats['估算token数']}, 总结={'是' if triggered else '否'}")
        
        # 如果触发了总结，显示摘要
        if triggered:
            print(f"   📋 生成摘要: {memory.historical_summary[:80]}...")
    
    print(f"\n【最终短期记忆上下文】")
    print("-" * 40)
    final_context = memory.get_context()
    print(final_context[:300] + "..." if len(final_context) > 300 else final_context)
    
    print(f"\n【最终统计】")
    print(memory.get_stats())
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)


if __name__ == "__main__":
    test_short_term_memory()