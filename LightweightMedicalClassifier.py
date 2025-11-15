import json
from typing import Dict, List, Tuple, Optional
from APIManager import APIManager

class LightweightMedicalClassifier:
    """
    一个轻量级的医疗相关性分类器（二级守门员）。
    用来过滤掉，聚类后的质点对应的上下文信息。 
    """
    
    def __init__(self, api_manager: APIManager = None):
        """
        初始化轻量级医疗分类器
        
        Args:
            api_manager: API管理器实例，如果为None则创建默认实例
        """
        # TODO: 未来将这里替换为本地SFT（监督微调）的小模型
        
        # 初始化API管理器
        self.api_manager = api_manager if api_manager else APIManager()
        if not self.api_manager.is_available():
            print("⚠️ 警告: API不可用，医疗分类功能将无法使用")



    def _build_system_prompt(self) -> str:
        """
        构建分类器的系统提示词 (System Prompt)。
        针对“詹姆斯问题”进行优化，明确区分“患者病历”与“第三方新闻”。
        """
        return """你是一个高效、轻量级的医疗相关性二元分类器。

你的唯一任务是判断输入的文本是否与**用户本人（患者）**的医疗、健康、疾病、症状或身体伤害**直接相关**。

## 核心原则
我们只关心用户本人（第一人称 "我"）或其咨询对象（例如 "我妈"）的健康状况。我们的目标是为医生过滤出“有价值的病历信息**”。

## 分类规则

### 相关的 (is_medical: true)
- 用户本人的症状 (例如: 头疼, 恶心, 睡不着, 焦虑)
- 用户本人的疾病 (例如: 高血压, 糖尿病)
- 用户本人的药物 (例如: 布洛芬)
- 用户本人的身体受伤 (例如: 崴脚, 骨折, 划伤)
- 家属相关的医疗咨询 (例如: "我妈有高血压...")
- 假设性的医疗咨询 (例如: "如果吃了头孢...")

### 不相关的 (is_medical: false)
- 纯粹的社交问候 (例如: 你好, 谢谢, 再见)
- 纯粹的闲聊 (例如: 天气很好, 你是机器人吗)
- APP技术问题 (例如: 网络卡, 怎么用)
- 第三方/名人新闻：提及与用户本人病历无关的第三方（如名人、新闻）的医疗事件。
- 纯粹比喻：明确的夸张或比喻 (例如: "气得我血压高")

## 关键边界示例
- "我喜欢看篮球" -> `{"is_medical": false}`
- "我打篮球受伤了" -> `{"is_medical": true}`
- "我妈有高血压" -> `{"is_medical": true}` (家属咨询，相关)
- "看篮球让我心脏都快跳出来了" -> `{"is_medical": false}` (比喻，无关)
- "我看到新闻说詹姆斯把脚崴了" -> `{"is_medical": false}` (第三方新闻，与患者病历无关)
- "我老板气得我血压都高了" -> `{"is_medical": false}` (夸张比喻，无关)
- "我感觉我的人生病了" -> `{"is_medical": false}` (纯粹哲学比喻)
- "我感觉我的人生病了，最近做什么都没劲" -> `{"is_medical": true}` (后半句是“乏力”症状)

## 输出格式
你必须只返回一个 JSON 对象，格式如下：
`{"is_medical": <true_or_false>}`
不要返回任何其他文本、寒暄或解释。
"""
    def classify(self, text: str) -> Optional[bool]:
        """
        对单句文本进行分类。
        
        Args:
            text: 需要分类的句子。
            
        Returns:
            - True: 医疗相关
            - False: 非医疗
            - None: API 调用或解析失败
        """
        if not self.api_manager.is_available():
            print(f"错误: API不可用 - 无法分类: '{text}'")
            return None

        # 使用API管理器进行调用
        user_prompt = f"请对以下文本进行分类：\n\n{text}"
        
        result = self.api_manager.call_json_completion(
            system_prompt=self._build_system_prompt(),
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=50
        )
        
        if not result:
            print(f"API调用或JSON解析失败")
            return None
        
        is_medical = result.get("is_medical")  # 不设默认值
        
        if isinstance(is_medical, bool):
            return is_medical
        else:
            print(f"警告: API返回了意外的非布尔值: {is_medical}")
            return None  # 视为解析失败


