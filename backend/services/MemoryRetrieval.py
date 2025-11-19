"""
长期记忆查询逻辑：从向量数据库和图数据库检索历史记忆
整合：RAGIntentClassifier + DatabaseManager + SentenceTransformer
"""
import json
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

from backend.core.DatabaseManager import get_db_manager
from backend.ml.RAGIntentClassifier import RAGIntentClassifier
from backend.core.config import settings


class MemoryRetrieval:
    """
    长期记忆检索管理器

    核心流程：
    1. 判断是否需要RAG（RAGIntentClassifier）
    2. 如果需要RAG：
       - 向量数据库语义检索（主要）
       - 图数据库关系查询（按需）
    3. 整合检索结果并格式化
    """

    def __init__(self):
        """初始化长期记忆检索器"""
        self.db = get_db_manager()
        self.rag_classifier = RAGIntentClassifier()
        self.sbert_model = SentenceTransformer(settings.sbert_model)

        print("✓ 长期记忆检索器初始化完成")

    def retrieve(
        self,
        patient_id: str,
        user_query: str,
        short_term_context: str,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        检索长期记忆

        Args:
            patient_id: 患者ID
            user_query: 用户查询
            short_term_context: 短期记忆上下文
            n_results: 返回结果数量

        Returns:
            {
                "need_rag": bool,
                "rag_triggered": bool,
                "vector_results": List[Dict],
                "graph_results": List[Dict] | None,
                "formatted_context": str
            }
        """
        print(f"\n开始检索长期记忆:")
        print(f"  患者ID: {patient_id}")
        print(f"  查询: {user_query[:50]}...")

        # 1. RAG意图分类 + 查询策略判断
        rag_result = self.rag_classifier.classify_with_strategy(
            user_query=user_query,
            short_term_context=short_term_context
        )

        if not rag_result:
            print("  ✗ RAG意图分类失败")
            return self._empty_result()

        need_rag = rag_result['need_rag']
        print(f"  need_rag: {need_rag}")
        print(f"  reason: {rag_result['reason']}")

        if not need_rag:
            print("  → 不需要RAG，使用短期记忆即可")
            return self._empty_result()

        # 2. 获取查询策略
        query_strategy = rag_result['query_strategy']
        print(f"  查询策略: 向量DB={query_strategy['vector_db']}, 图DB={query_strategy['graph_db']}")

        # 3. 向量数据库检索
        vector_results = []
        if query_strategy['vector_db']:
            vector_results = self._retrieve_from_vector_db(
                patient_id=patient_id,
                user_query=user_query,
                n_results=n_results
            )
            print(f"  ✓ 向量检索完成: 找到 {len(vector_results)} 条记忆")

        # 4. 图数据库检索（按需）
        graph_results = None
        if query_strategy['graph_db']:
            graph_query_type = query_strategy['graph_query_type']
            graph_results = self._retrieve_from_graph_db(
                patient_id=patient_id,
                query_type=graph_query_type
            )
            print(f"  ✓ 图检索完成: 找到 {len(graph_results) if graph_results else 0} 条记录")

        # 5. 格式化结果
        formatted_context = self._format_retrieval_results(
            vector_results=vector_results,
            graph_results=graph_results
        )

        return {
            "need_rag": need_rag,
            "rag_triggered": True,
            "confidence": rag_result['confidence'],
            "reason": rag_result['reason'],
            "query_strategy": query_strategy,
            "vector_results": vector_results,
            "graph_results": graph_results,
            "formatted_context": formatted_context
        }

    def _retrieve_from_vector_db(
        self,
        patient_id: str,
        user_query: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        从向量数据库检索

        Returns:
            [
                {
                    "unit_id": str,
                    "session_id": str,
                    "unit_type": str,
                    "similarity": float,
                    "narrative_summary": str,
                    "session_topic": str,
                    "knowledge_graph": dict,
                    "dialogue_rounds": int
                },
                ...
            ]
        """
        # 1. 生成查询向量
        query_embedding = self.sbert_model.encode(user_query).tolist()

        # 2. 查询Chroma
        chroma_results = self.db.query_memory_by_vector(
            query_embedding=query_embedding,
            patient_id=patient_id,
            n_results=n_results
        )

        # 3. 解析结果
        formatted_results = []

        if not chroma_results or not chroma_results['ids']:
            return formatted_results

        ids = chroma_results['ids'][0]
        metadatas = chroma_results['metadatas'][0]
        documents = chroma_results['documents'][0]
        distances = chroma_results.get('distances', [[]])[0]

        for i, unit_id in enumerate(ids):
            metadata = metadatas[i]
            document = documents[i]
            distance = distances[i] if i < len(distances) else 0.0

            # 解析完整分析结果
            analysis = json.loads(metadata['analysis_json'])

            formatted_results.append({
                "unit_id": unit_id,
                "session_id": metadata['session_id'],
                "unit_type": metadata['unit_type'],
                "cluster_id": metadata.get('cluster_id'),
                "similarity": 1 - distance,  # 转换为相似度
                "created_at": metadata['created_at'],
                "narrative_summary": document,
                "session_topic": analysis.get('session_topic', ''),
                "main_complaint": analysis.get('main_complaint_vectorized', ''),
                "knowledge_graph": analysis.get('knowledge_graph', {}),
                "dialogue_rounds": analysis.get('dialogue_rounds', 0)
            })

        return formatted_results

    def _retrieve_from_graph_db(
        self,
        patient_id: str,
        query_type: str
    ) -> Optional[List[Dict]]:
        """
        从图数据库检索

        Args:
            patient_id: 患者ID
            query_type: 查询类型（drug_interaction, symptom_disease等）

        Returns:
            图查询结果列表
        """
        try:
            results = self.db.query_graph(
                query_type=query_type,
                patient_id=patient_id
            )
            return results

        except Exception as e:
            print(f"  ⚠️  图查询失败: {e}")
            return None

    def _format_retrieval_results(
        self,
        vector_results: List[Dict],
        graph_results: Optional[List[Dict]]
    ) -> str:
        """
        格式化检索结果为文本上下文

        用于提供给LLM生成回复时使用

        Returns:
            格式化的上下文字符串
        """
        context_parts = []

        # 1. 向量检索结果
        if vector_results:
            context_parts.append("【历史医疗记录】\n")

            for i, result in enumerate(vector_results, 1):
                session_topic = result['session_topic']
                summary = result['narrative_summary']
                created_at = result['created_at'][:10]  # 只取日期部分

                context_parts.append(
                    f"{i}. [{created_at}] {session_topic}\n"
                    f"   摘要: {summary}\n"
                )

        # 2. 图检索结果
        if graph_results:
            context_parts.append("\n【相关医疗信息】\n")

            # 根据不同的查询类型格式化
            if isinstance(graph_results, list) and graph_results:
                first_result = graph_results[0]

                # 药物相互作用
                if 'drug1' in first_result and 'drug2' in first_result:
                    context_parts.append("药物相互作用警告:\n")
                    for item in graph_results:
                        severity = item.get('severity', '未知')
                        context_parts.append(
                            f"  - {item['drug1']} + {item['drug2']}: "
                            f"{severity}级别相互作用\n"
                        )

                # 症状-疾病关联
                elif 'symptom' in first_result and 'disease' in first_result:
                    context_parts.append("症状可能关联的疾病:\n")
                    for item in graph_results:
                        context_parts.append(
                            f"  - 症状「{item['symptom']}」可能是「{item['disease']}」\n"
                        )

                # 治疗历史
                elif 'drug' in first_result and 'prescribed_at' in first_result:
                    context_parts.append("历史用药记录:\n")
                    for item in graph_results:
                        date = item['prescribed_at'][:10]
                        dosage = item.get('dosage', '未记录')
                        context_parts.append(
                            f"  - [{date}] {item['drug']} ({dosage})\n"
                        )

        return "".join(context_parts) if context_parts else ""

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        """返回空结果"""
        return {
            "need_rag": False,
            "rag_triggered": False,
            "vector_results": [],
            "graph_results": None,
            "formatted_context": ""
        }


# 全局检索管理器实例
_memory_retrieval: Optional[MemoryRetrieval] = None


def get_memory_retrieval() -> MemoryRetrieval:
    """获取全局检索管理器实例（单例模式）"""
    global _memory_retrieval
    if _memory_retrieval is None:
        _memory_retrieval = MemoryRetrieval()
    return _memory_retrieval


# 单元测试
if __name__ == "__main__":
    print("=== 长期记忆检索测试 ===\n")

    retrieval = MemoryRetrieval()

    # 测试查询
    print("【测试1 - 需要RAG的查询】")
    result = retrieval.retrieve(
        patient_id="P12345",
        user_query="上次医生给我开的降压药还能继续吃吗？",
        short_term_context="",
        n_results=3
    )

    print(f"\n检索结果:")
    print(f"  need_rag: {result['need_rag']}")
    if result['rag_triggered']:
        print(f"  置信度: {result.get('confidence', 0):.2f}")
        print(f"  原因: {result.get('reason', '')}")
        print(f"  向量结果数: {len(result['vector_results'])}")
        print(f"  图结果: {'有' if result['graph_results'] else '无'}")
        print(f"\n格式化上下文:")
        print(result['formatted_context'][:200] if result['formatted_context'] else "(空)")

    print("\n测试完成")
