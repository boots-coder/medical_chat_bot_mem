"""
测试数据：为短期记忆管理器、RAG意图分类器、最终回复生成器提供测试场景

包含：
1. 需要RAG的完整对话场景
2. 不需要RAG的完整对话场景  
3. 预设的长期记忆数据（用于模拟RAG检索结果）
"""

# 测试场景1：需要RAG的对话
# 用户在当前session中提到了历史症状和上次就诊，需要查找长期记忆
RAG_REQUIRED_CONVERSATION = [
    {"role": "user", "content": "你好医生，我又头痛了"},
    {"role": "assistant", "content": "您好，请问您的头痛是什么样的？什么部位疼痛？"},
    {"role": "user", "content": "还是太阳穴那里，一阵一阵的疼"},
    {"role": "assistant", "content": "明白了。请问您之前有过类似的头痛吗？"},
    {"role": "user", "content": "有的，和上次来看病时的症状很像，但是这次感觉更严重一些"},
    # 这里用户提到了"上次来看病时"，需要RAG检索历史记录
    {"role": "assistant", "content": "我需要查看一下您之前的就诊记录，以便为您提供更准确的建议"},
    {"role": "user", "content": "好的，我想知道这次和上次比起来，我需要调整用药吗？上次开的布洛芬还能继续吃吗？"}
    # 最后这个query明确需要RAG：对比历史症状 + 历史用药
]

# 测试场景2：不需要RAG的对话
# 完全是新症状描述和一般性医疗咨询，当前session上下文足够回答
NO_RAG_CONVERSATION = [
    {"role": "user", "content": "医生您好，我今天突然感觉胃痛"},
    {"role": "assistant", "content": "您好，请描述一下胃痛的具体情况，是什么时候开始的？"},
    {"role": "user", "content": "大概两个小时前开始的，就是上腹部这里痛，有点胀"},
    {"role": "assistant", "content": "请问您今天吃了什么？有没有恶心呕吐的症状？"},
    {"role": "user", "content": "中午吃得比较油腻，有点恶心但没有呕吐"},
    {"role": "assistant", "content": "听起来可能是消化不良。建议您可以先少吃清淡易消化的食物，可以服用一些健胃消食片"},
    {"role": "user", "content": "健胃消食片怎么吃？饭前还是饭后？"}
    # 最后这个query不需要RAG：询问的是当前session刚提到的药物用法
]

# 预设的长期记忆数据（模拟RAG检索结果）
# 这些数据模拟从向量数据库、图数据库、SQL数据库检索到的历史信息
MOCK_LONG_TERM_MEMORY = {
    "vector_search_results": [
        {
            "session_id": "sess_202410_001",
            "date": "2024-10-15",
            "summary": "用户主诉偏头痛，太阳穴位置阵发性疼痛，伴恶心畏光。诊断为偏头痛，建议布洛芬400mg tid，效果良好。",
            "similarity_score": 0.95
        },
        {
            "session_id": "sess_202409_003", 
            "date": "2024-09-28",
            "summary": "用户头痛复发，症状与前次相似但较轻。继续布洛芬治疗，建议注意休息规律。",
            "similarity_score": 0.88
        }
    ],
    "graph_search_results": [
        {
            "entity_type": "Patient",
            "entity_id": "user_789", 
            "relationships": [
                {"relation": "HAS_SYMPTOM", "target": "S_Headache", "status": "recurring"},
                {"relation": "DIAGNOSED_WITH", "target": "DG_Migraine", "confidence": "high"},
                {"relation": "PRESCRIBED", "target": "DR_Ibuprofen", "dosage": "400mg tid", "last_prescribed": "2024-10-15"}
            ]
        }
    ],
    "sql_search_results": [
        {
            "session_id": "sess_202410_001",
            "user_id": "user_789",
            "start_time": "2024-10-15T14:30:00Z",
            "end_time": "2024-10-15T14:45:00Z",
            "session_topic": "偏头痛复诊及用药调整",
            "dialogue_rounds": 8
        }
    ]
}

# 综合的RAG检索结果（格式化后的长期记忆上下文）
FORMATTED_LONG_TERM_CONTEXT = """【历史就诊记录】
2024年10月15日：用户主诉偏头痛，太阳穴位置阵发性疼痛，伴恶心畏光。诊断为偏头痛，处方布洛芬400mg每日三次，治疗效果良好。

2024年9月28日：头痛复发，症状与前次相似但程度较轻，继续布洛芬治疗。

【用药历史】
布洛芬 400mg，每日三次，最后处方时间：2024年10月15日，患者反馈效果良好。

【诊断记录】
偏头痛（反复发作），症状表现为太阳穴阵发性疼痛，常伴恶心畏光。"""


def get_test_scenario(scenario_type: str):
    """
    获取测试场景数据
    
    Args:
        scenario_type: "rag_required" 或 "no_rag"
        
    Returns:
        测试场景字典，包含对话数据和相关信息
    """
    if scenario_type == "rag_required":
        return {
            "name": "需要RAG的对话场景",
            "description": "用户提到历史症状对比和历史用药，需要查找长期记忆",
            "conversation": RAG_REQUIRED_CONVERSATION,
            "expected_rag": True,
            "final_query": RAG_REQUIRED_CONVERSATION[-1]["content"],
            "mock_rag_result": FORMATTED_LONG_TERM_CONTEXT
        }
    
    elif scenario_type == "no_rag":
        return {
            "name": "不需要RAG的对话场景",
            "description": "新症状咨询，当前session上下文足够回答",
            "conversation": NO_RAG_CONVERSATION,
            "expected_rag": False,
            "final_query": NO_RAG_CONVERSATION[-1]["content"],
            "mock_rag_result": ""  # 不需要RAG，无长期记忆
        }
    
    else:
        raise ValueError(f"未知的场景类型: {scenario_type}")


def test_data_integration():
    """测试数据完整性检查"""
    print("="*60)
    print("测试数据完整性检查")
    print("="*60)
    
    # 测试RAG场景
    rag_scenario = get_test_scenario("rag_required")
    print(f"\n【{rag_scenario['name']}】")
    print(f"描述: {rag_scenario['description']}")
    print(f"对话轮数: {len(rag_scenario['conversation'])}")
    print(f"最终查询: {rag_scenario['final_query']}")
    print(f"期望RAG: {rag_scenario['expected_rag']}")
    print(f"模拟RAG结果长度: {len(rag_scenario['mock_rag_result'])}字符")
    
    # 测试非RAG场景
    no_rag_scenario = get_test_scenario("no_rag")
    print(f"\n【{no_rag_scenario['name']}】")
    print(f"描述: {no_rag_scenario['description']}")
    print(f"对话轮数: {len(no_rag_scenario['conversation'])}")
    print(f"最终查询: {no_rag_scenario['final_query']}")
    print(f"期望RAG: {no_rag_scenario['expected_rag']}")
    
    # 显示长期记忆数据结构
    print(f"\n【模拟长期记忆数据】")
    print(f"向量检索结果: {len(MOCK_LONG_TERM_MEMORY['vector_search_results'])}条")
    print(f"图检索结果: {len(MOCK_LONG_TERM_MEMORY['graph_search_results'])}条")
    print(f"SQL检索结果: {len(MOCK_LONG_TERM_MEMORY['sql_search_results'])}条")
    
    print(f"\n【格式化的长期记忆上下文预览】")
    print(FORMATTED_LONG_TERM_CONTEXT[:200] + "...")
    
    print("\n" + "="*60)
    print("数据检查完成")
    print("="*60)


if __name__ == "__main__":
    test_data_integration()