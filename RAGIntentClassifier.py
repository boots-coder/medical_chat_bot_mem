import json
from typing import Optional
from APIManager import APIManager


class RAGIntentClassifier:
    """
    RAG意图分类器：判断用户查询是否需要检索长期记忆
    
    与ShortTermMemoryManager集成，基于当前query和短期记忆上下文
    判断是否需要触发跨session的RAG查询
    
    前提条件：用户来访次数 > 1（已有历史记录）
    """
    
    def __init__(self, api_manager: APIManager = None):
        """
        初始化RAG意图分类器
        
        Args:
            api_manager: API管理器实例，如果为None则创建默认实例
        """
        # 初始化API管理器
        self.api_manager = api_manager if api_manager else APIManager()
        if not self.api_manager.is_available():
            print("⚠️ 警告: API不可用，RAG意图分类功能将无法使用")

    def _build_system_prompt(self) -> str:
        """构建RAG意图分类的系统提示词"""
        return """你是一个专业的RAG意图分类器，专门为医疗对话系统设计。

你的任务是判断用户的查询是否需要检索**长期记忆**（跨session历史记录）。

## 背景说明
- 短期记忆：当前session内的对话上下文（包括历史摘要+当前窗口），始终可用
- 长期记忆：跨session的历史医疗记录，需要RAG检索，成本较高
- 系统会向你提供：用户当前查询 + 完整的短期记忆上下文

## 分类原则

### 需要RAG检索 (需要长期记忆) - 返回 true
1. **明确的历史引用**：
   - "上次医生说的..."、"之前检查的结果..."
   - "以前也有过..."、"我的病史中..."
   - "上一次来看病时..."

2. **症状对比和趋势**：
   - "这次和上次的头痛比..."
   - "比之前严重了/好转了"
   - "又犯了同样的病"、"老毛病又犯了"

3. **治疗效果跟踪**：
   - "吃了上次开的药后..."
   - "按照之前的建议..."
   - "上次的治疗方案效果..."

4. **慢性病管理询问**：
   - "我的高血压控制得怎么样"
   - "糖尿病最近的变化"
   - "定期复查的结果对比"

### 不需要RAG检索 (短期记忆足够) - 返回 false
1. **新症状描述**：
   - 首次提及的症状或感受
   - 当前的身体状况描述

2. **一般医学咨询**：
   - 通用疾病知识询问
   - 药物使用方法

3. **当前对话延续**：
   - 对刚才提及内容的进一步询问
   - 短期记忆上下文中已包含足够信息

## 关键判断点
- 如果用户使用"上次"、"之前"、"以前"等时间词汇，且**不是**指当前session内容 → 需要RAG
- 如果短期记忆上下文中已包含用户询问的相关信息 → 不需要RAG
- 如果是症状的历史对比或趋势分析 → 需要RAG

## 输出格式
你必须只返回一个JSON对象：
```json
{
  "need_rag": true/false,
  "confidence": 0.0-1.0,
  "reason": "简短说明判断依据"
}
```

注意：
- need_rag: true表示需要检索长期记忆，false表示短期记忆足够
- confidence: 置信度分数 (0.0-1.0)
- reason: 一句话说明判断的核心依据"""

    def classify_rag_intent(self, user_query: str, short_term_context: str) -> Optional[dict]:
        """
        分类用户查询的RAG意图
        
        Args:
            user_query: 用户当前的查询内容
            short_term_context: 短期记忆管理器提供的上下文
            
        Returns:
            {
                "need_rag": bool,        # 是否需要RAG检索
                "confidence": float,     # 置信度 0.0-1.0
                "reason": str           # 判断原因
            }
            返回None表示API调用失败
        """
        if not self.api_manager.is_available():
            print(f"错误: API不可用 - 无法分类: '{user_query}'")
            return None

        # 构建用户提示词
        if short_term_context.strip():
            user_prompt = f"""用户当前查询：{user_query}

短期记忆上下文：
{short_term_context}

请根据用户查询和短期记忆上下文，判断是否需要检索长期记忆。"""
        else:
            user_prompt = f"""用户当前查询：{user_query}

短期记忆上下文：（空，新会话开始）

请判断是否需要检索长期记忆。"""
        
        # 使用API管理器进行调用
        result = self.api_manager.call_json_completion(
            system_prompt=self._build_system_prompt(),
            user_prompt=user_prompt,
            temperature=0.1,  # 保证分类一致性
            max_tokens=200
        )
        
        if not result:
            print(f"RAG意图分类API调用失败")
            return None
        
        # 验证返回格式
        required_keys = ["need_rag", "confidence", "reason"]
        if not all(key in result for key in required_keys):
            print(f"警告: API返回格式不完整: {result}")
            return None
        
        # 验证数据类型
        if not isinstance(result["need_rag"], bool):
            print(f"警告: need_rag 不是布尔值: {result['need_rag']}")
            return None
            
        if not isinstance(result["confidence"], (int, float)) or not (0 <= result["confidence"] <= 1):
            print(f"警告: confidence 不在 0-1 范围内: {result['confidence']}")
            return None
        
        return result

    def quick_check(self, user_query: str, short_term_context: str = "") -> bool:
        """
        快速检查是否需要RAG（简化版本，仅返回True/False）
        
        Args:
            user_query: 用户查询
            short_term_context: 短期记忆上下文
            
        Returns:
            True: 需要RAG检索, False: 不需要RAG检索或API调用失败
        """
        result = self.classify_rag_intent(user_query, short_term_context)
        return result["need_rag"] if result else False


# 单元测试
def test_rag_intent_classifier():
    """RAG意图分类器单元测试（与短期记忆管理器集成）"""
    print("="*60)
    print("RAG意图分类器单元测试")
    print("="*60)
    
    classifier = RAGIntentClassifier()
    
    # 测试用例1：需要RAG - 历史症状对比
    print("\n【测试用例1 - 需要RAG】")
    print("场景：用户在讨论当前头痛时，提到了历史对比")
    print("-" * 50)
    
    short_context_1 = """历史摘要：用户主诉头痛三天，太阳穴位置阵发性疼痛，伴有恶心。诊断可能为偏头痛，建议服用布洛芬。

当前对话：
用户：我按照您的建议吃了布洛芬，感觉好一些了
助手：很好，症状有缓解是好现象。请继续观察"""
    
    query_1 = "这次的头痛和我上次来看的时候比起来，是不是严重一些？"
    
    result_1 = classifier.classify_rag_intent(query_1, short_context_1)
    if result_1:
        status = "✓" if result_1["need_rag"] else "✗"
        print(f"{status} 查询: {query_1}")
        print(f"   结果: need_rag={result_1['need_rag']}, confidence={result_1['confidence']:.2f}")
        print(f"   原因: {result_1['reason']}")
        print(f"   上下文: {short_context_1[:100]}...")
    
    # 测试用例2：不需要RAG - 当前session内容充足
    print("\n【测试用例2 - 不需要RAG】")  
    print("场景：用户询问的内容在短期记忆中已包含")
    print("-" * 50)
    
    short_context_2 = """当前对话：
用户：我胃痛，想吃点药
助手：建议您服用奥美拉唑，饭前半小时服用，一天一次
用户：好的，谢谢医生"""
    
    query_2 = "刚才您说的奥美拉唑，我需要吃多久？"
    
    result_2 = classifier.classify_rag_intent(query_2, short_context_2)
    if result_2:
        status = "✓" if not result_2["need_rag"] else "✗"
        print(f"{status} 查询: {query_2}")
        print(f"   结果: need_rag={result_2['need_rag']}, confidence={result_2['confidence']:.2f}")
        print(f"   原因: {result_2['reason']}")
        print(f"   上下文: {short_context_2}")
    
    # 测试用例3：边界情况 - 空上下文但提及历史
    print("\n【测试用例3 - 边界情况：空上下文但提及历史】")
    print("-" * 50)
    
    query_3 = "上次医生给我开的高血压药还能继续吃吗？"
    short_context_3 = ""
    
    result_3 = classifier.classify_rag_intent(query_3, short_context_3)
    if result_3:
        status = "✓" if result_3["need_rag"] else "✗"
        print(f"{status} 查询: {query_3}")
        print(f"   结果: need_rag={result_3['need_rag']}, confidence={result_3['confidence']:.2f}")
        print(f"   原因: {result_3['reason']}")
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)


if __name__ == "__main__":
    test_rag_intent_classifier()