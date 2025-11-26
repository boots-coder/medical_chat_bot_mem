"""
RAG Intent Classification Training Data Schema
训练数据结构定义和样本模板
"""
from typing import List, Dict, Any
from pydantic import BaseModel, Field


class RAGIntentSample(BaseModel):
    """单个RAG意图分类样本"""
    query: str = Field(..., description="用户查询内容")
    short_term_context: str = Field(..., description="短期记忆上下文")
    need_rag: bool = Field(..., description="是否需要RAG检索")
    confidence: float = Field(..., ge=0.0, le=1.0, description="标注置信度")
    reason: str = Field(..., description="分类理由")
    category: str = Field(..., description="样本所属的类别标签")


class RAGIntentDataset(BaseModel):
    """完整的RAG意图分类数据集"""
    samples: List[RAGIntentSample]
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ==================== 数据生成模板 ====================

# 定义需要RAG的场景类别（中英文混合）
NEED_RAG_CATEGORIES = {
    "历史症状比较": {
        "description": "用户将当前症状与历史就诊进行比较 / User compares current symptoms with previous visits",
        "keywords_zh": ["上次", "之前", "以前", "比上次", "和之前相比", "还是老样子"],
        "keywords_en": ["last time", "previous", "before", "compared to", "than last", "same as"],
        "examples_zh": [
            "这次头痛比上次严重吗?",
            "和之前的胃痛比起来怎么样?",
            "我的血压比上个月的检查结果如何?"
        ],
        "examples_en": [
            "Is this headache worse than last time?",
            "How does it compare to my previous stomach pain?",
            "How is my blood pressure compared to last month's results?"
        ]
    },
    "历史诊疗查询": {
        "description": "询问过往的诊断、检查或治疗记录 / Asking about previous diagnosis, tests or treatment records",
        "keywords_zh": ["上次医生说", "之前检查", "上次开的药", "以前的诊断", "历史记录"],
        "keywords_en": ["doctor said last time", "previous test", "prescribed before", "past diagnosis", "medical history"],
        "examples_zh": [
            "上次医生说我是什么病来着?",
            "之前的血常规结果是什么?",
            "我能继续吃上次开的降压药吗?"
        ],
        "examples_en": [
            "What did the doctor say was wrong last time?",
            "What were my previous blood test results?",
            "Can I continue taking the blood pressure medication from last visit?"
        ]
    },
    "用药历史追溯": {
        "description": "询问过往用药情况和效果 / Asking about previous medications and their effectiveness",
        "keywords_zh": ["上次开的", "之前吃的", "以前用过", "上回的药", "那个药效果"],
        "keywords_en": ["medication from last time", "previously took", "used before", "that medicine", "drug effect"],
        "examples_zh": [
            "上次开的布洛芬还有吗，我能继续吃吗?",
            "之前吃的胃药叫什么名字?",
            "上次那个止痛药效果怎么样?"
        ],
        "examples_en": [
            "Can I continue taking the ibuprofen prescribed last time?",
            "What was the name of the stomach medicine I took before?",
            "How effective was that painkiller from last visit?"
        ]
    },
    "慢病管理询问": {
        "description": "慢性病患者询问病情控制情况或趋势 / Chronic disease patients asking about condition control or trends",
        "keywords_zh": ["我的高血压", "糖尿病控制", "最近血糖", "病情变化", "指标趋势"],
        "keywords_en": ["my hypertension", "diabetes control", "recent blood sugar", "condition changes", "indicator trends"],
        "examples_zh": [
            "我的高血压最近控制得怎么样?",
            "糖尿病这几个月有好转吗?",
            "我的肝功能指标趋势如何?"
        ],
        "examples_en": [
            "How well controlled is my hypertension recently?",
            "Has my diabetes improved over the past few months?",
            "What's the trend in my liver function indicators?"
        ]
    },
    "治疗效果跟踪": {
        "description": "询问治疗方案的效果或进展 / Asking about treatment effectiveness or progress",
        "keywords_zh": ["治疗效果", "有没有好转", "康复情况", "恢复得", "疗效"],
        "keywords_en": ["treatment effect", "any improvement", "recovery status", "getting better", "efficacy"],
        "examples_zh": [
            "上次的治疗方案效果如何?",
            "我的关节炎有好转吗?",
            "康复训练做得怎么样了?"
        ],
        "examples_en": [
            "How effective was the treatment plan from last time?",
            "Has my arthritis improved?",
            "How is my rehabilitation training going?"
        ]
    },
    "复发症状识别": {
        "description": "用户描述症状可能是历史疾病复发 / User describes symptoms that might be disease recurrence",
        "keywords_zh": ["又", "还是", "老毛病", "又犯了", "反复", "复发"],
        "keywords_en": ["again", "back", "old problem", "recurring", "relapse", "flare up"],
        "examples_zh": [
            "我的偏头痛又犯了",
            "胃又开始疼了，和之前一样",
            "老毛病又来了，该怎么办?"
        ],
        "examples_en": [
            "My migraine is back again",
            "My stomach is hurting again, just like before",
            "The old problem is back, what should I do?"
        ]
    }
}

# 定义不需要RAG的场景类别（中英文混合）
NO_NEED_RAG_CATEGORIES = {
    "新症状描述": {
        "description": "首次描述当前症状或感受 / First-time description of current symptoms",
        "keywords_zh": ["我现在", "今天", "刚刚", "突然", "最近"],
        "keywords_en": ["right now", "today", "just now", "suddenly", "recently"],
        "examples_zh": [
            "我今天头很痛",
            "刚才吃完饭就开始胃疼",
            "最近两天一直咳嗽"
        ],
        "examples_en": [
            "I have a severe headache today",
            "My stomach started hurting right after eating",
            "I've been coughing for the past two days"
        ]
    },
    "当前对话延续": {
        "description": "针对刚才讨论的内容进行追问 / Follow-up on current conversation",
        "keywords_zh": ["刚才说的", "您提到的", "这个", "那个", "什么意思"],
        "keywords_en": ["you just said", "you mentioned", "this", "that", "what do you mean"],
        "examples_zh": [
            "刚才您说的布洛芬怎么吃?",
            "您提到的注意事项具体是什么?",
            "这个药有副作用吗?"
        ],
        "examples_en": [
            "How should I take the ibuprofen you just mentioned?",
            "What exactly are the precautions you mentioned?",
            "Does this medication have side effects?"
        ]
    },
    "通用医疗咨询": {
        "description": "询问一般医学知识，无需历史记录 / General medical knowledge questions",
        "keywords_zh": ["什么是", "怎么预防", "为什么会", "一般", "通常"],
        "keywords_en": ["what is", "how to prevent", "why does", "generally", "usually"],
        "examples_zh": [
            "什么是高血压?",
            "感冒怎么预防?",
            "为什么会头痛?"
        ],
        "examples_en": [
            "What is hypertension?",
            "How can I prevent colds?",
            "Why do I get headaches?"
        ]
    },
    "用药方法询问": {
        "description": "询问当前讨论药物的用法用量 / Asking about current medication usage",
        "keywords_zh": ["怎么吃", "怎么用", "一天几次", "饭前还是饭后", "用量"],
        "keywords_en": ["how to take", "how to use", "times per day", "before or after meals", "dosage"],
        "examples_zh": [
            "布洛芬怎么吃?",
            "这个药一天吃几次?",
            "是饭前吃还是饭后吃?"
        ],
        "examples_en": [
            "How should I take ibuprofen?",
            "How many times a day should I take this medicine?",
            "Should I take it before or after meals?"
        ]
    },
    "简单确认问题": {
        "description": "对刚才信息的确认或重复 / Simple confirmations",
        "keywords_zh": ["好的", "明白了", "知道了", "谢谢", "懂了"],
        "keywords_en": ["okay", "got it", "understand", "thank you", "I see"],
        "examples_zh": [
            "好的，我知道了",
            "明白了，谢谢医生",
            "懂了，我会注意的"
        ],
        "examples_en": [
            "Okay, I understand",
            "Got it, thank you doctor",
            "I see, I'll be careful"
        ]
    },
    "当前状态询问": {
        "description": "询问当前症状的处理方法，无历史对比 / Asking about current symptom management",
        "keywords_zh": ["怎么办", "需要", "应该", "可以", "能不能"],
        "keywords_en": ["what should I do", "need to", "should I", "can I", "is it okay"],
        "examples_zh": [
            "我现在头痛怎么办?",
            "需要去医院吗?",
            "可以吃点什么药?"
        ],
        "examples_en": [
            "What should I do about my headache?",
            "Do I need to go to the hospital?",
            "What medication can I take?"
        ]
    }
}


# ==================== 数据生成提示词模板 ====================

DATA_GENERATION_SYSTEM_PROMPT = """你是一个医疗对话数据标注专家，专门为RAG意图分类器生成训练样本。

## 任务说明
生成高质量的RAG意图分类训练样本。每个样本包含：
1. query: 用户的查询内容
2. short_term_context: 短期记忆上下文（可能为空）
3. need_rag: 是否需要RAG检索（true/false）
4. confidence: 你对该标注的置信度（0.0-1.0）
5. reason: 分类理由（简短说明）
6. category: 样本所属的类别标签

## 生成原则
1. **多样性**: 同一类别下生成多种表述方式
2. **真实性**: 模拟真实用户的口语化表达，包括错别字、语气词
3. **难度分层**:
   - 简单样本（80%）: 明确的需要/不需要RAG
   - 困难样本（20%）: 边界情况，需要结合上下文判断
4. **上下文变化**:
   - 50%的样本有短期记忆上下文
   - 50%的样本上下文为空
5. **平衡性**: need_rag=true 和 false 的样本各占50%

## 输出格式
返回JSON数组，每个元素是一个样本对象：
```json
[
  {
    "query": "用户查询",
    "short_term_context": "历史摘要: ...\n当前对话:\n...",
    "need_rag": true,
    "confidence": 0.95,
    "reason": "用户提到'上次'，且短期记忆中无相关信息",
    "category": "历史症状比较"
  }
]
```

## 医疗场景特殊要求
1. 涵盖常见疾病：头痛、高血压、糖尿病、感冒、胃病、关节炎等
2. 包含中文医学术语和口语化表达
3. 模拟不同年龄段用户的表达习惯
4. 考虑慢性病管理的长期跟踪场景
"""


def build_data_generation_prompt(
    category_type: str,
    category_name: str,
    num_samples: int = 10,
    difficulty: str = "mixed"
) -> str:
    """
    构建数据生成的用户提示词

    Args:
        category_type: "need_rag" 或 "no_need_rag"
        category_name: 类别名称
        num_samples: 生成样本数量
        difficulty: "easy", "hard", "mixed"
    """
    if category_type == "need_rag":
        category_info = NEED_RAG_CATEGORIES.get(category_name)
        need_rag = True
    else:
        category_info = NO_NEED_RAG_CATEGORIES.get(category_name)
        need_rag = False

    if not category_info:
        raise ValueError(f"未知类别: {category_name}")

    difficulty_desc = {
        "easy": "生成明确、易于分类的样本",
        "hard": "生成边界情况、需要仔细分析的困难样本",
        "mixed": "80%简单样本 + 20%困难样本"
    }

    prompt = f"""请为以下类别生成 {num_samples} 个训练样本：

【类别信息】
- 类别名称: {category_name}
- 是否需要RAG: {need_rag}
- 描述: {category_info['description']}
- 关键词: {', '.join(category_info['keywords'])}

【参考示例】
{chr(10).join(f"  - {ex}" for ex in category_info['examples'])}

【难度要求】
{difficulty_desc[difficulty]}

【特殊要求】
1. 生成真实的短期记忆上下文（约50%样本）
2. 模拟口语化表达，包括语气词、停顿
3. 确保 confidence 分数合理（简单样本0.9+，困难样本0.6-0.8）
4. reason 要简洁明确，指出关键判断依据

请严格按照JSON格式返回样本数组。"""

    return prompt


# ==================== 工具函数 ====================

def get_all_categories() -> Dict[str, List[str]]:
    """获取所有类别名称"""
    return {
        "need_rag": list(NEED_RAG_CATEGORIES.keys()),
        "no_need_rag": list(NO_NEED_RAG_CATEGORIES.keys())
    }


def get_category_distribution() -> Dict[str, int]:
    """推荐的类别样本数量分布"""
    return {
        # 需要RAG的类别（每类50-80样本）
        "历史症状比较": 80,
        "历史诊疗查询": 70,
        "用药历史追溯": 60,
        "慢病管理询问": 50,
        "治疗效果跟踪": 50,
        "复发症状识别": 40,

        # 不需要RAG的类别（每类50-80样本）
        "新症状描述": 80,
        "当前对话延续": 70,
        "通用医疗咨询": 60,
        "用药方法询问": 50,
        "简单确认问题": 40,
        "当前状态询问": 50,

        # 总计: 约700样本
    }


if __name__ == "__main__":
    print("="*60)
    print("RAG意图分类训练数据Schema")
    print("="*60)

    print("\n【需要RAG的类别】")
    for cat, info in NEED_RAG_CATEGORIES.items():
        print(f"\n{cat}:")
        print(f"  描述: {info['description']}")
        print(f"  示例: {info['examples'][0]}")

    print("\n【不需要RAG的类别】")
    for cat, info in NO_NEED_RAG_CATEGORIES.items():
        print(f"\n{cat}:")
        print(f"  描述: {info['description']}")
        print(f"  示例: {info['examples'][0]}")

    print("\n【推荐样本分布】")
    distribution = get_category_distribution()
    need_rag_total = sum(v for k, v in distribution.items() if k in NEED_RAG_CATEGORIES)
    no_need_rag_total = sum(v for k, v in distribution.items() if k in NO_NEED_RAG_CATEGORIES)
    print(f"  需要RAG: {need_rag_total} 样本")
    print(f"  不需要RAG: {no_need_rag_total} 样本")
    print(f"  总计: {need_rag_total + no_need_rag_total} 样本")
