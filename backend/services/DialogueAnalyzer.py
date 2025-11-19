from typing import List, Dict
import json
from backend.ml.APIManager import APIManager


class TokenLimitExceeded(Exception):
    """对话token数超出限制异常"""
    pass


class DialogueAnalyzer:
    """
    对话分析器：将医疗对话信息转换为三层存储结构，
    用于后面的存储；提示词工程。
    后续可以进行小模型微调替换。 
    """
    
    def __init__(
        self, 
        api_manager: APIManager = None,
        max_input_tokens: int = 4096
    ):
        """
        初始化对话分析器
        
        Args:
            api_manager: API管理器实例，如果为None则创建默认实例
            max_input_tokens: 最大输入token数限制
        """
        self.api_manager = api_manager if api_manager else APIManager()
        self.max_input_tokens = max_input_tokens
        
        if not self.api_manager.is_available():
            print("⚠️ 警告: API不可用，对话分析功能将无法使用")
        
    def analyze_session(
        self, 
        dialogue_list: List[Dict[str, str]],
        session_id: str,
        user_id: str,
        start_time: str,  
        end_time: str     
    ) -> Dict:
        """
        分析对话session，返回三层存储结构
        
        Args:
            dialogue_list: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            包含SQLite、VectorDB、GraphDB数据的字典
        """
        # TODO: 实现token计数逻辑
        # estimated_tokens = self._estimate_tokens(dialogue_list)
        # if estimated_tokens > self.max_input_tokens:
        #     raise TokenLimitExceeded(f"对话tokens({estimated_tokens})超出限制({self.max_input_tokens})")
        
        # 构建提示词
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(dialogue_list, session_id, user_id,start_time, end_time)
        
        # 使用API管理器调用LLM
        if not self.api_manager.is_available():
            raise RuntimeError("API不可用，无法进行对话分析")
        
        result = self.api_manager.call_json_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,  # 低温度确保结构化输出的一致性
            max_tokens=3000  # 增加输出长度以容纳完整的知识图谱
        )
        
        if not result:
            raise RuntimeError("对话分析API调用失败")
        
        return result
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return """你是一个专业的医疗对话分析助手，负责将医患对话转换为结构化的三层数据存储格式。

## 任务目标
分析医疗对话，提取信息并输出JSON格式，包含三个层次：
1. **SQLite（事实层）**：会话元数据（时间、轮次、主题）
2. **VectorDB（语义层）**：对话摘要和主诉的语义化表达
3. **GraphDB（关系层）**：医学实体及其关系的知识图谱

## 输出格式规范
```json
{
  "session_id": "会话ID (必须来自用户提示)",
  "user_id": "用户ID (必须来自用户提示)",
  "start_time": "ISO8601格式开始时间 (必须来自用户提示)",
  "end_time": "ISO8601格式结束时间 (必须来自用户提示)",
  "dialogue_rounds": 对话轮次数(整数, 必须来自用户提示)",
  "session_topic": "一句话概括对话主题",
  
  "narrative_summary": "3-5句话完整叙述对话过程，包含主诉、评估、建议",
  "main_complaint_vectorized": "提取用户主诉的核心关键词组合",
  
  "knowledge_graph": {
    "entities": [
      {"id": "唯一标识", "type": "实体类型", "label": "实体名称"}
    ],
    "relationships": [
      {"subject": "主体实体ID", "predicate": "关系类型", "object": "客体实体ID"}
    ]
  }
}
```

## 实体类型规范
- **Patient**: 患者（使用user_id作为ID）
- **Symptom**: 症状（ID前缀：`S_`，例如 `S_Headache`）
- **Disease**: 疾病（ID前缀：`D_`，例如 `D_Hypertension`）
- **Diagnosis**: 诊断结论（ID前缀：`DG_`，例如 `DG_TensionHeadache`）
- **Drug**: 药物（ID前缀：`DR_`，例如 `DR_Ibuprofen`）
- **Examination**: 检查项目（ID前缀：`E_`，例如 `E_BloodPressure`）
- **Treatment**: 治疗方案（ID前缀：`T_`，例如 `T_BedRest`）

## 关系类型规范
- **HAS_SYMPTOM**: 患者有症状
- **HAS_HISTORY**: 患者有病史
- **MAY_CAUSE**: 疾病可能导致症状
- **IS_SUGGESTED_FOR**: 诊断建议给患者
- **RECOMMENDED_FOR**: 药物/治疗推荐给诊断
- **REQUIRES**: 诊断需要检查
- **INDICATES**: 检查结果指示疾病

## 注意事项
1. 实体ID使用英文，label使用中文
2. 时间戳使用ISO 8601格式（UTC）
3. 所有实体ID必须标准化（未来对接ICD-10、SNOMED CT等医学编码库）
4. 关系必须准确反映医学逻辑
5. 必须返回合法的JSON格式
6. 引用完整性：在 'relationships' 列表中引用的**每一个** `subject` 和 `object` 的 `id`，都**必须**在 `entities` 列表中被**提前声明**。

请严格按照上述格式输出。"""

    def _build_user_prompt(
        self, 
        dialogue_list: List[Dict[str, str]],
        session_id: str,
        user_id: str,
        start_time: str,  # <-- [修改点 1] 接收参数
        end_time: str     # <-- [修改点 2] 接收参数
    ) -> str:
        """构建用户提示词"""
        # 格式化对话历史
        dialogue_text = ""
        for i, turn in enumerate(dialogue_list, 1):
            role = "用户" if turn["role"] == "user" else "助手"
            dialogue_text += f"[第{i}轮 - {role}]: {turn['content']}\n\n"
        
        # [修改点 3] 在 Python 端计算真实的轮数 (一问一答 = 1轮)
        # 注意：这里用 `len(dialogue_list)` 是总轮次(turns)，除以2才是轮数(rounds)
        dialogue_rounds = max(1, len(dialogue_list) // 2) # 避免为0
        
        # [修改点 4] 将所有元数据注入到 f-string 中
        return f"""请分析以下医疗对话，提取结构化信息。

                ## 会话信息 (请直接复制这些信息到JSON中，不要修改)
                - session_id: {session_id}
                - user_id: {user_id}
                - start_time: {start_time}
                - end_time: {end_time}
                - dialogue_rounds: {dialogue_rounds}

                ## 对话内容
                {dialogue_text}

                请按照系统提示中的JSON格式输出分析结果。"""
    
    def _estimate_tokens(self, dialogue_list: List[Dict[str, str]]) -> int:
        """
        估算对话的token数量
        
        TODO: 使用tiktoken库精确计数
        简单实现：中文1字≈1.5tokens，英文1词≈1.3tokens
        """
        total_chars = sum(len(turn["content"]) for turn in dialogue_list)
        estimated_tokens = int(total_chars * 1.5)  # 粗略估算
        return estimated_tokens


# 使用示例
if __name__ == "__main__":
    analyzer = DialogueAnalyzer(max_input_tokens=4096)

    # 模拟对话数据
    dialogue = [
        {"role": "user", "content": "你好，我最近两天一直头痛，还有点恶心"},
        {"role": "assistant", "content": "您好，请问您有高血压或其他慢性病史吗？"},
        {"role": "user", "content": "有高血压，已经三年了"},
        {"role": "assistant", "content": "头痛可能与血压升高有关。建议监测血压，必要时可服用布洛芬缓解症状"}
    ]
    
    try:
        result = analyzer.analyze_session(
            dialogue_list=dialogue,
            session_id="sess_1a2b3c",
            user_id="user_789",
            start_time="2025-11-04T12:30:00Z",  
            end_time="2025-11-04T12:34:51Z"    
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (TokenLimitExceeded, RuntimeError) as e:
        print(f"错误: {e}")
        # TODO: 未来实现对话压缩逻辑
        # compressed_dialogue = compress_dialogue(dialogue)
        # result = analyzer.analyze_session(compressed_dialogue, ...)