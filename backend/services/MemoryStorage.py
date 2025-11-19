"""
Long-term memory storage logic: Long-term memory storage at the end of a session
Integration: Clusterer + Classifier + DialogueAnalyzer + DatabaseManager
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
    Long-term memory storage manager

    Core workflow:
    1. Determine if clustering is needed (based on dialogue length)
    2. If clustering is needed: filter each cluster for medical relevance
    3. Analyze each medically relevant unit with DialogueAnalyzer
    4. Store to Chroma (vector database)
    5. Store to Neo4j (knowledge graph)
    """

    def __init__(self):
        """Initialize long-term memory storage"""
        self.db = get_db_manager()
        self.dialogue_analyzer = DialogueAnalyzer()
        self.medical_classifier = LightweightMedicalClassifier()
        self.clusterer = ContextAwareDialogueClusterer(
            min_cluster_size=settings.cluster_min_size,
            min_samples=settings.cluster_min_samples
        )
        self.sbert_model = SentenceTransformer(settings.sbert_model)

        print("✓ Long-term memory storage initialized")

    def store_session_memory(
        self,
        session_id: str,
        patient_id: str,
        dialogue_list: List[Dict[str, str]],
        start_time: str,
        end_time: str
    ):
        """
        Store long-term memory for a session

        Args:
            session_id: Session ID
            patient_id: Patient ID
            dialogue_list: Dialogue list [{"role": "user", "content": "..."}, ...]
            start_time: Session start time (ISO 8601 format)
            end_time: Session end time (ISO 8601 format)
        """
        print(f"\nStarting to store session long-term memory: {session_id}")
        print(f"  Dialogue turns: {len(dialogue_list)}")
        print(f"  Patient ID: {patient_id}")

        # 1. Determine if clustering is needed
        dialogue_turns = len(dialogue_list)
        needs_clustering = dialogue_turns > settings.max_dialogue_turns

        if needs_clustering:
            print(f"  Dialogue exceeds {settings.max_dialogue_turns} turns, triggering clustering")
            self._store_with_clustering(
                session_id, patient_id, dialogue_list, start_time, end_time
            )
        else:
            print(f"  Dialogue is short, storing directly")
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
        Store a single session (without clustering)

        Workflow:
        1. Analyze entire session using DialogueAnalyzer
        2. Store to Chroma
        3. Store to Neo4j
        """
        print("\n[Single Session Storage Workflow]")

        # 1. Dialogue analysis
        print("  Step 1: Analyzing dialogue...")
        try:
            analysis = self.dialogue_analyzer.analyze_session(
                dialogue_list=dialogue_list,
                session_id=session_id,
                user_id=patient_id,
                start_time=start_time,
                end_time=end_time
            )
        except Exception as e:
            print(f"  ✗ Dialogue analysis failed: {e}")
            return

        print(f"  ✓ Analysis complete: {analysis['session_topic']}")

        # 2. Store to vector database
        self._store_to_vector_db(
            unit_id=session_id,
            unit_type="session",
            session_id=session_id,
            patient_id=patient_id,
            cluster_id=None,
            analysis=analysis
        )

        # 3. Store to graph database
        self._store_to_graph_db(
            patient_id=patient_id,
            session_id=session_id,
            knowledge_graph=analysis['knowledge_graph'],
            timestamp=end_time
        )

        print(f"  ✓ Single session storage complete: {session_id}")

    def _store_with_clustering(
        self,
        session_id: str,
        patient_id: str,
        dialogue_list: List[Dict[str, str]],
        start_time: str,
        end_time: str
    ):
        """
        Store after clustering

        Workflow:
        1. Cluster using ContextAwareDialogueClusterer
        2. Filter each cluster for medical relevance
        3. Analyze and store medically relevant clusters separately
        """
        print("\n[Clustering Storage Workflow]")

        # 1. Clustering
        print("  Step 1: Clustering dialogues...")
        try:
            cluster_result = self.clusterer.process(
                dialogue_list=dialogue_list,
                return_details=True
            )
        except Exception as e:
            print(f"  ✗ Clustering failed: {e}")
            return

        labels = cluster_result['labels']
        qa_pairs = cluster_result['qa_pairs']
        cluster_dialogues = cluster_result['cluster_dialogues']
        details = cluster_result.get('details', {})

        print(f"  ✓ Clustering complete:")
        print(f"    - Number of valid clusters: {details.get('n_clusters', 0)}")
        print(f"    - Number of noise points: {details.get('n_noise_points', 0)}")

        # 2. Process each cluster
        print("\n  Step 2: Filtering and storing each cluster...")
        stored_count = 0

        for cluster_id, qa_list in cluster_dialogues.items():
            print(f"\n    Processing cluster {cluster_id} ({len(qa_list)} QA pairs):")

            # Noise points need individual classification and storage
            if cluster_id == -1:
                print(f"      → Noise cluster, evaluating medical relevance individually...")
                noise_stored = 0

                for idx, qa in enumerate(qa_list):
                    # Medical relevance determination for individual QA pair
                    single_text = self._format_qa_pairs_to_text([qa])
                    is_medical = self.medical_classifier.classify(single_text)

                    if is_medical is None:
                        print(f"        Noise point {idx}: ⚠️ Classifier failed, skipping")
                        continue

                    if not is_medical:
                        print(f"        Noise point {idx}: ✗ Not medically relevant, skipping")
                        continue

                    print(f"        Noise point {idx}: ✓ Medically relevant, starting analysis...")

                    # Convert to dialogue format
                    single_dialogue = self._qa_pairs_to_dialogue([qa])

                    # Dialogue analysis
                    try:
                        analysis = self.dialogue_analyzer.analyze_session(
                            dialogue_list=single_dialogue,
                            session_id=session_id,
                            user_id=patient_id,
                            start_time=start_time,
                            end_time=end_time
                        )
                    except Exception as e:
                        print(f"        Noise point {idx}: ✗ Analysis failed: {e}")
                        continue

                    # Store individual noise point
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
                    print(f"        Noise point {idx}: ✓ Storage complete")

                stored_count += noise_stored
                print(f"      Noise cluster processing complete: stored {noise_stored}/{len(qa_list)} QA pairs")

            else:
                # Valid cluster: overall evaluation and storage
                # 2.1 Convert to dialogue format
                cluster_dialogue = self._qa_pairs_to_dialogue(qa_list)

                # 2.2 Medical relevance filtering
                cluster_text = self._format_qa_pairs_to_text(qa_list)
                is_medical = self.medical_classifier.classify(cluster_text)

                if is_medical is None:
                    print(f"      ⚠️  Classifier failed, skipping")
                    continue

                if not is_medical:
                    print(f"      ✗ Not medically relevant, skipping")
                    continue

                print(f"      ✓ Medically relevant, starting analysis...")

                # 2.3 Dialogue analysis
                try:
                    analysis = self.dialogue_analyzer.analyze_session(
                        dialogue_list=cluster_dialogue,
                        session_id=session_id,
                        user_id=patient_id,
                        start_time=start_time,
                        end_time=end_time
                    )
                except Exception as e:
                    print(f"      ✗ Analysis failed: {e}")
                    continue

                # 2.4 Storage
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
                print(f"      ✓ Cluster {cluster_id} storage complete")

        print(f"\n  ✓ Clustering storage complete: stored {stored_count} units in total")

    def _store_to_vector_db(
        self,
        unit_id: str,
        unit_type: str,
        session_id: str,
        patient_id: str,
        cluster_id: int,
        analysis: Dict[str, Any]
    ):
        """Store to vector database (Chroma)"""
        try:
            # 1. Generate embedding
            narrative_summary = analysis['narrative_summary']
            embedding = self.sbert_model.encode(narrative_summary).tolist()

            # 2. Build metadata (Chroma doesn't accept None values)
            metadata = {
                "patient_id": patient_id,
                "unit_type": unit_type,
                "session_id": session_id,
                "created_at": analysis['start_time'],
                "end_time": analysis['end_time'],
                "analysis_json": json.dumps(analysis, ensure_ascii=False)
            }

            # Only add cluster_id for clustering results
            if cluster_id is not None:
                metadata["cluster_id"] = cluster_id

            # 3. Store
            self.db.store_memory_unit(
                unit_id=unit_id,
                embedding=embedding,
                document=narrative_summary,
                metadata=metadata
            )

        except Exception as e:
            print(f"      ✗ Vector storage failed: {e}")
            raise

    def _store_to_graph_db(
        self,
        patient_id: str,
        session_id: str,
        knowledge_graph: Dict[str, Any],
        timestamp: str
    ):
        """Store to graph database (Neo4j)"""
        try:
            self.db.store_knowledge_graph(
                patient_id=patient_id,
                session_id=session_id,
                knowledge_graph=knowledge_graph,
                timestamp=timestamp
            )
        except Exception as e:
            print(f"      ⚠️  Graph storage failed (non-critical): {e}")
            # Graph storage failure doesn't affect main workflow

    @staticmethod
    def _qa_pairs_to_dialogue(qa_pairs: List[Dict]) -> List[Dict[str, str]]:
        """Convert QA pair list to dialogue list"""
        dialogue = []
        for qa in qa_pairs:
            dialogue.append({"role": "user", "content": qa['user']})
            dialogue.append({"role": "assistant", "content": qa['assistant']})
        return dialogue

    @staticmethod
    def _format_qa_pairs_to_text(qa_pairs: List[Dict]) -> str:
        """Format QA pairs as text (for classifier)"""
        texts = []
        for qa in qa_pairs:
            texts.append(f"User: {qa['user']}")
            texts.append(f"Assistant: {qa['assistant']}")
        return "\n".join(texts)


# Global storage manager instance
_memory_storage: Optional[MemoryStorage] = None


def get_memory_storage() -> MemoryStorage:
    """Get global storage manager instance (singleton pattern)"""
    global _memory_storage
    if _memory_storage is None:
        _memory_storage = MemoryStorage()
    return _memory_storage


# Unit tests
if __name__ == "__main__":
    print("=== Long-term Memory Storage Test ===\n")

    storage = MemoryStorage()

    # Test data: short dialogue
    short_dialogue = [
        {"role": "user", "content": "I've been having severe headaches lately"},
        {"role": "assistant", "content": "When did your headaches start?"},
        {"role": "user", "content": "About three days ago"},
        {"role": "assistant", "content": "I suggest you take ibuprofen to relieve the symptoms"}
    ]

    print("[Test 1 - Short Dialogue Storage]")
    storage.store_session_memory(
        session_id="TEST_S001",
        patient_id="P_TEST_001",
        dialogue_list=short_dialogue,
        start_time="2025-11-19T10:00:00Z",
        end_time="2025-11-19T10:05:00Z"
    )

    print("\nTest complete")
