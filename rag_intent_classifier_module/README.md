# RAG意图分类器模块

本地轻量级RAG意图分类器，用于快速判断用户查询是否需要检索长期记忆。

## 核心功能

- **快速推理**: <50ms per query (CPU)
- **双语支持**: 中文 + 英文
- **高精度**: 测试准确率 98.94%
- **本地运行**: 无需API调用

## 目录结构

```
rag_intent_classifier_module/
├── training/
│   ├── local_classifier.py      # 分类器实现
│   └── data_schema.py           # 数据schema定义
├── data/                         # 训练/测试数据
│   ├── train_data.json          # 752样本
│   └── test_data.json           # 188样本
├── models/final/                 # 训练好的模型（1.1GB）
├── results/                      # 测试结果
├── train.py                      # 训练脚本
└── test.py                       # 测试脚本
```

## 快速使用

### 1. 加载模型

```python
from rag_intent_classifier_module.training.local_classifier import LocalRAGIntentClassifier

# 加载训练好的模型
classifier = LocalRAGIntentClassifier.load_model("rag_intent_classifier_module/models/final")
```

### 2. 预测

```python
# 单次预测
result = classifier.predict(
    query="上次医生开的降压药我能继续吃吗?",
    short_term_context=""
)

print(f"Need RAG: {result['need_rag']}")       # True
print(f"Confidence: {result['confidence']:.3f}") # 0.998
```

### 3. 集成示例

```python
def should_retrieve_memory(query: str, context: str = "") -> bool:
    """判断是否需要检索长期记忆"""
    result = classifier.predict(query=query, short_term_context=context)
    return result['need_rag']

# 在对话流程中使用
user_query = "我的血压比上个月怎么样?"
if should_retrieve_memory(user_query):
    # 触发RAG检索
    long_term_memory = retrieve_from_vectordb(user_query)
else:
    # 直接使用短期记忆
    long_term_memory = None
```

## 性能指标

- **准确率**: 98.94% (186/188)
- **推理速度**: ~30ms per query (M1 CPU)
- **模型大小**: 1.1GB
- **基础模型**: xlm-roberta-base

## 分类规则

### 需要RAG（检索长期记忆）
- 历史症状比较："这次头痛比上次严重吗?"
- 历史诊疗查询："上次医生说我是什么病?"
- 用药历史追溯："之前吃的降压药叫什么?"
- 慢病管理询问："我的糖尿病最近控制得怎么样?"
- 治疗效果跟踪："上次的治疗方案效果如何?"
- 复发症状识别："我的偏头痛又犯了"

### 不需要RAG（仅用短期记忆）
- 新症状描述："我今天头很痛"
- 当前对话延续："刚才您说的布洛芬怎么吃?"
- 通用医疗咨询："什么是高血压?"
- 用药方法询问："这个药一天吃几次?"
- 当前状态询问："我现在头痛怎么办?"
- 简单确认问题："好的，我知道了"

## 训练（可选）

如需重新训练模型：

```bash
python train.py \
  --train-data data/train_data.json \
  --test-data data/test_data.json \
  --model xlm-roberta-base \
  --epochs 5 \
  --batch-size 16 \
  --device mps \
  --output-dir models/final
```

## 测试

```bash
# 批量测试
python test.py \
  --model-dir models/final \
  --test-file data/test_data.json \
  --output results/test_results.json

# 交互式测试
python test.py --model-dir models/final --interactive
```

## 依赖

- torch
- transformers
- scikit-learn
- numpy
- tqdm
- pydantic

## 集成步骤

1. **导入分类器**
   ```python
   from rag_intent_classifier_module.training.local_classifier import LocalRAGIntentClassifier
   ```

2. **初始化模型**（应用启动时）
   ```python
   rag_classifier = LocalRAGIntentClassifier.load_model("rag_intent_classifier_module/models/final")
   ```

3. **替换原有LLM调用**
   ```python
   # 旧代码：调用LLM判断是否需要RAG（慢）
   # need_rag = llm.classify_intent(query)

   # 新代码：使用本地分类器（快）
   result = rag_classifier.predict(query=query, short_term_context=context)
   need_rag = result['need_rag']
   ```

4. **性能提升**
   - 延迟：从 ~2000ms 降至 ~30ms（约67倍提升）
   - 成本：从API调用变为本地推理（零成本）
   - 准确性：98.94%

## 注意事项

- 模型文件较大（1.1GB），首次加载需10-15秒
- 建议在应用启动时加载一次，后续复用
- 支持CPU/GPU/MPS，自动检测最优设备
- protobuf警告可忽略，不影响功能

---

**版本**: v1.0
**最后更新**: 2025-11-27
**状态**: 已训练并测试，可直接使用
