# 智能医疗咨询系统

基于LLM的医疗对话系统，具备短期记忆、长期记忆和RAG检索功能。

## 功能特性

- ✅ **会话管理**: JWT token验证、30分钟超时机制
- ✅ **短期记忆**: 当前会话的对话上下文管理和自动总结
- ✅ **长期记忆**: 跨会话的历史记录存储（向量DB + 图DB）
- ✅ **智能聚类**: 超长对话自动聚类和医疗相关性过滤
- ✅ **RAG检索**: 智能判断何时需要检索历史记忆
- ✅ **实时对话**: WebSocket实时通信
- ✅ **三层数据库**: SQLite(会话) + Chroma(向量) + Neo4j(知识图谱)

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

已提供 `.env` 文件，主要配置：
- API密钥已配置
- 数据库路径已设置
- JWT密钥已生成

### 3. 启动Neo4j (可选)

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/changeme \
  neo4j:latest
```

### 4. 启动服务

```bash
python main.py
```

访问：http://localhost:8000/docs

## 使用流程

```
外部医疗系统 → 创建会话 → 生成URL token → 患者点击链接 →
实时对话 → 结束会话 → 长期记忆存储 → 外部系统获取摘要
```

## 核心模块

| 模块 | 功能 |
|-----|------|
| TokenManager | JWT token生成和验证 |
| SessionManager | 会话创建、验证、超时管理 |
| MemoryStorage | 长期记忆存储（聚类+分析） |
| MemoryRetrieval | 长期记忆检索（RAG） |
| DatabaseManager | 三数据库统一管理 |

## API接口

- `POST /api/external/create-session` - 创建会话
- `GET /chat/{token}` - 聊天界面
- `WebSocket /ws/{session_id}` - 实时对话
- `POST /api/session/{session_id}/end` - 结束会话
- `GET /api/session/{session_id}/summary` - 获取摘要

详细文档：http://localhost:8000/docs
