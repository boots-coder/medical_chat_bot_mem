"""
长期记忆存储逻辑：会话结束时的长期记忆存储
整合：Clusterer + Classifier + DialogueAnalyzer + DatabaseManager
"""
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer

from backend.core.DatabaseManager import get_db_manager
from backend.services.DialogueAnalyzer import DialogueAnalyzer
from backend.ml.LightweightMedicalClassifier import LightweightMedicalClassifier
from backend.ml.context_aware_clusterer import ContextAwareDialogueClusterer
from backend.core.config import settings


class MemoryStorage:
    """
    长期记忆存储管理器

    核心流程：
    1. 判断是否需要聚类（根据对话长度）
    2. 如果需要聚类：对每个簇进行医疗相关性过滤
    3. 对每个医疗相关的单元进行DialogueAnalyzer分析
    4. 存储到Chroma（向量数据库）
    5. 存储到Neo4j（知识图谱）
    """

    def __init__(self):
        """初始化长期记忆存储器"""
        self.db = get_db_manager()
        self.dialogue_analyzer = DialogueAnalyzer()
        self.medical_classifier = LightweightMedicalClassifier()
        self.clusterer = ContextAwareDialogueClusterer(
            min_cluster_size=settings.cluster_min_size,
            min_samples=settings.cluster_min_samples
        )
        self.sbert_model = SentenceTransformer(settings.sbert_model)

        print("✓ 长期记忆存储器初始化完成")

    def store_session_memory(
        self,
        session_id: str,
        patient_id: str,
        dialogue_list: List[Dict[str, str]],
        start_time: str,
        end_time: str
    ):
        """
        存储会话的长期记忆

        Args:
            session_id: 会话ID
            patient_id: 患者ID
            dialogue_list: 对话列表 [{"role": "user", "content": "..."}, ...]
            start_time: 会话开始时间（ISO 8601格式）
            end_time: 会话结束时间（ISO 8601格式）
        """
        print(f"\n开始存储会话长期记忆: {session_id}")
        print(f"  对话轮数: {len(dialogue_list)}")
        print(f"  患者ID: {patient_id}")

        # 1. 判断是否需要聚类
        dialogue_turns = len(dialogue_list)
        needs_clustering = dialogue_turns > settings.max_dialogue_turns

        if needs_clustering:
            print(f"  对话超过{settings.max_dialogue_turns}轮，触发聚类")
            self._store_with_clustering(
                session_id, patient_id, dialogue_list, start_time, end_time
            )
        else:
            print(f"  对话较短，直接存储")
            self._store_single_session(
                session_id, patient_id, dialogue_list, start_time, end_time
            )

    def _store_single_session(
        self,
        session_id: str,
        patient_id: str,
        dialogue_list: List[Dict[str, str]],
        start_time: str,
        end_time: str
    ):
        """
        存储单个会话（不聚类）

        流程：
        1. 使用DialogueAnalyzer分析整个会话
        2. 存储到Chroma
        3. 存储到Neo4j
        """
        print("\n【单会话存储流程】")

        # 1. 对话分析
        print("  步骤1: 分析对话...")
        try:
            analysis = self.dialogue_analyzer.analyze_session(
                dialogue_list=dialogue_list,
                session_id=session_id,
                user_id=patient_id,
                start_time=start_time,
                end_time=end_time
            )
        except Exception as e:
            print(f"  ✗ 对话分析失败: {e}")
            return

        print(f"  ✓ 分析完成: {analysis['session_topic']}")

        # 2. 存储到向量数据库
        self._store_to_vector_db(
            unit_id=session_id,
            unit_type="session",
            session_id=session_id,
            patient_id=patient_id,
            cluster_id=None,
            analysis=analysis
        )

        # 3. 存储到图数据库
        self._store_to_graph_db(
            patient_id=patient_id,
            session_id=session_id,
            knowledge_graph=analysis['knowledge_graph'],
            timestamp=end_time
        )

        print(f"  ✓ 单会话存储完成: {session_id}")

    def _store_with_clustering(
        self,
        session_id: str,
        patient_id: str,
        dialogue_list: List[Dict[str, str]],
        start_time: str,
        end_time: str
    ):
        """
        聚类后存储

        流程：
        1. 使用ContextAwareDialogueClusterer聚类
        2. 对每个簇进行医疗相关性过滤
        3. 对医疗相关的簇分别分析和存储
        """
        print("\n【聚类存储流程】")

        # 1. 聚类
        print("  步骤1: 对话聚类...")
        try:
            cluster_result = self.clusterer.process(
                dialogue_list=dialogue_list,
                return_details=True
            )
        except Exception as e:
            print(f"  ✗ 聚类失败: {e}")
            return

        labels = cluster_result['labels']
        qa_pairs = cluster_result['qa_pairs']
        cluster_dialogues = cluster_result['cluster_dialogues']
        details = cluster_result.get('details', {})

        print(f"  ✓ 聚类完成:")
        print(f"    - 有效簇数: {details.get('n_clusters', 0)}")
        print(f"    - 噪声点数: {details.get('n_noise_points', 0)}")

        # 2. 对每个簇进行处理
        print("\n  步骤2: 过滤和存储每个簇...")
        stored_count = 0

        for cluster_id, qa_list in cluster_dialogues.items():
            print(f"\n    处理簇 {cluster_id} ({len(qa_list)}个问答对):")

            # 噪声点需要逐个分类和存储
            if cluster_id == -1:
                print(f"      → 噪声簇，逐个判断医疗相关性...")
                noise_stored = 0

                for idx, qa in enumerate(qa_list):
                    # 单个问答对的医疗相关性判断
                    single_text = self._format_qa_pairs_to_text([qa])
                    is_medical = self.medical_classifier.classify(single_text)

                    if is_medical is None:
                        print(f"        噪声点{idx}: ⚠️ 分类器失败，跳过")
                        continue

                    if not is_medical:
                        print(f"        噪声点{idx}: ✗ 非医疗相关，跳过")
                        continue

                    print(f"        噪声点{idx}: ✓ 医疗相关，开始分析...")

                    # 转换为dialogue格式
                    single_dialogue = self._qa_pairs_to_dialogue([qa])

                    # 对话分析
                    try:
                        analysis = self.dialogue_analyzer.analyze_session(
                            dialogue_list=single_dialogue,
                            session_id=session_id,
                            user_id=patient_id,
                            start_time=start_time,
                            end_time=end_time
                        )
                    except Exception as e:
                        print(f"        噪声点{idx}: ✗ 分析失败: {e}")
                        continue

                    # 存储单个噪声点
                    unit_id = f"{session_id}_noise_{idx}"
                    self._store_to_vector_db(
                        unit_id=unit_id,
                        unit_type="noise",
                        session_id=session_id,
                        patient_id=patient_id,
                        cluster_id=-1,
                        analysis=analysis
                    )

                    self._store_to_graph_db(
                        patient_id=patient_id,
                        session_id=session_id,
                        knowledge_graph=analysis['knowledge_graph'],
                        timestamp=end_time
                    )

                    noise_stored += 1
                    print(f"        噪声点{idx}: ✓ 存储完成")

                stored_count += noise_stored
                print(f"      噪声簇处理完成: 存储了 {noise_stored}/{len(qa_list)} 个问答对")

            else:
                # 有效簇：整体判断和存储
                # 2.1 转换为dialogue格式
                cluster_dialogue = self._qa_pairs_to_dialogue(qa_list)

                # 2.2 医疗相关性过滤
                cluster_text = self._format_qa_pairs_to_text(qa_list)
                is_medical = self.medical_classifier.classify(cluster_text)

                if is_medical is None:
                    print(f"      ⚠️  分类器失败，跳过")
                    continue

                if not is_medical:
                    print(f"      ✗ 非医疗相关，跳过")
                    continue

                print(f"      ✓ 医疗相关，开始分析...")

                # 2.3 对话分析
                try:
                    analysis = self.dialogue_analyzer.analyze_session(
                        dialogue_list=cluster_dialogue,
                        session_id=session_id,
                        user_id=patient_id,
                        start_time=start_time,
                        end_time=end_time
                    )
                except Exception as e:
                    print(f"      ✗ 分析失败: {e}")
                    continue

                # 2.4 存储
                unit_id = f"{session_id}_cluster_{cluster_id}"
                self._store_to_vector_db(
                    unit_id=unit_id,
                    unit_type="cluster",
                    session_id=session_id,
                    patient_id=patient_id,
                    cluster_id=cluster_id,
                    analysis=analysis
                )

                self._store_to_graph_db(
                    patient_id=patient_id,
                    session_id=session_id,
                    knowledge_graph=analysis['knowledge_graph'],
                    timestamp=end_time
                )

                stored_count += 1
                print(f"      ✓ 簇 {cluster_id} 存储完成")

        print(f"\n  ✓ 聚类存储完成: 共存储 {stored_count} 个单元")

    def _store_to_vector_db(
        self,
        unit_id: str,
        unit_type: str,
        session_id: str,
        patient_id: str,
        cluster_id: int,
        analysis: Dict[str, Any]
    ):
        """存储到向量数据库（Chroma）"""
        try:
            # 1. 生成向量
            narrative_summary = analysis['narrative_summary']
            embedding = self.sbert_model.encode(narrative_summary).tolist()

            # 2. 构建metadata（Chroma不接受None值）
            metadata = {
                "patient_id": patient_id,
                "unit_type": unit_type,
                "session_id": session_id,
                "created_at": analysis['start_time'],
                "end_time": analysis['end_time'],
                "analysis_json": json.dumps(analysis, ensure_ascii=False)
            }

            # 只有聚类结果才添加 cluster_id
            if cluster_id is not None:
                metadata["cluster_id"] = cluster_id

            # 3. 存储
            self.db.store_memory_unit(
                unit_id=unit_id,
                embedding=embedding,
                document=narrative_summary,
                metadata=metadata
            )

        except Exception as e:
            print(f"      ✗ 向量存储失败: {e}")
            raise

    def _store_to_graph_db(
        self,
        patient_id: str,
        session_id: str,
        knowledge_graph: Dict[str, Any],
        timestamp: str
    ):
        """存储到图数据库（Neo4j）"""
        try:
            self.db.store_knowledge_graph(
                patient_id=patient_id,
                session_id=session_id,
                knowledge_graph=knowledge_graph,
                timestamp=timestamp
            )
        except Exception as e:
            print(f"      ⚠️  图存储失败（非关键）: {e}")
            # 图存储失败不影响主流程

    @staticmethod
    def _qa_pairs_to_dialogue(qa_pairs: List[Dict]) -> List[Dict[str, str]]:
        """将问答对列表转换为对话列表"""
        dialogue = []
        for qa in qa_pairs:
            dialogue.append({"role": "user", "content": qa['user']})
            dialogue.append({"role": "assistant", "content": qa['assistant']})
        return dialogue

    @staticmethod
    def _format_qa_pairs_to_text(qa_pairs: List[Dict]) -> str:
        """将问答对格式化为文本（用于分类器）"""
        texts = []
        for qa in qa_pairs:
            texts.append(f"用户：{qa['user']}")
            texts.append(f"助手：{qa['assistant']}")
        return "\n".join(texts)


# 全局存储管理器实例
_memory_storage: Optional[MemoryStorage] = None


def get_memory_storage() -> MemoryStorage:
    """获取全局存储管理器实例（单例模式）"""
    global _memory_storage
    if _memory_storage is None:
        _memory_storage = MemoryStorage()
    return _memory_storage


# 单元测试
if __name__ == "__main__":
    print("=== 长期记忆存储测试 ===\n")

    storage = MemoryStorage()

    # 测试数据：短对话
    short_dialogue = [
        {"role": "user", "content": "我最近头痛很厉害"},
        {"role": "assistant", "content": "请问您的头痛是什么时候开始的？"},
        {"role": "user", "content": "大概三天前"},
        {"role": "assistant", "content": "建议您服用布洛芬缓解症状"}
    ]

    print("【测试1 - 短对话存储】")
    storage.store_session_memory(
        session_id="TEST_S001",
        patient_id="P_TEST_001",
        dialogue_list=short_dialogue,
        start_time="2025-11-19T10:00:00Z",
        end_time="2025-11-19T10:05:00Z"
    )

    print("\n测试完成")
