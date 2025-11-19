#!/usr/bin/env python3
"""
检查 Chroma 数据库中存储的长期记忆
"""
import sys
sys.path.insert(0, '/Users/bootscoder/Desktop/medical_chat_memory_manager')

from backend.core.DatabaseManager import get_db_manager
import json

# 获取用户输入的 session_id 或查看所有
if len(sys.argv) > 1:
    session_id = sys.argv[1]
    print(f"查询会话: {session_id}")
    print("="*60)

    db = get_db_manager()
    results = db.chroma_collection.get(
        where={"session_id": session_id},
        include=["metadatas", "documents"]
    )

    if results['ids']:
        metadata = results['metadatas'][0]
        document = results['documents'][0]
        analysis = json.loads(metadata['analysis_json'])

        print(f"✓ 找到长期记忆！")
        print(f"\n【对话主题】")
        print(analysis.get('session_topic', 'N/A'))
        print(f"\n【完整摘要】")
        print(analysis.get('narrative_summary', 'N/A'))
        print(f"\n【主诉】")
        print(analysis.get('main_complaint_vectorized', 'N/A'))
        print(f"\n【对话轮数】")
        print(analysis.get('dialogue_rounds', 0))
    else:
        print(f"✗ 未找到该会话的长期记忆")
        print(f"\n可能原因:")
        print(f"1. 会话尚未结束")
        print(f"2. 长期记忆存储失败")
        print(f"3. 会话ID不正确")
else:
    # 列出所有存储的会话
    print("查询所有长期记忆...")
    print("="*60)

    db = get_db_manager()
    results = db.chroma_collection.get(
        include=["metadatas"]
    )

    if results['ids']:
        print(f"找到 {len(results['ids'])} 条长期记忆记录:\n")
        for i, doc_id in enumerate(results['ids']):
            metadata = results['metadatas'][i]
            analysis = json.loads(metadata['analysis_json'])
            print(f"{i+1}. Session: {metadata['session_id']}")
            print(f"   Patient: {metadata['patient_id']}")
            print(f"   Topic: {analysis.get('session_topic', 'N/A')}")
            print(f"   Time: {metadata['created_at']}")
            print()
    else:
        print("✗ 数据库中没有任何长期记忆记录")

print("="*60)
