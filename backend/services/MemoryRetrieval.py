"""
Long-term memory retrieval logic: Retrieve historical memories from vector database and graph database
Integration: RAGIntentClassifier + DatabaseManager + SentenceTransformer
"""
import json
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

from backend.core.DatabaseManager import get_db_manager
from backend.ml.RAGIntentClassifier import RAGIntentClassifier
from backend.core.config import settings


class MemoryRetrieval:
    """
    Long-term Memory Retrieval Manager

    Core workflow:
    1. Determine if RAG is needed (RAGIntentClassifier)
    2. If RAG is needed:
       - Vector database semantic search (primary)
       - Graph database relationship query (as needed)
    3. Integrate retrieval results and format them
    """

    def __init__(self):
        """Initialize long-term memory retrieval manager"""
        self.db = get_db_manager()
        self.rag_classifier = RAGIntentClassifier()
        self.sbert_model = SentenceTransformer(settings.sbert_model)

        print("✓ Long-term memory retrieval manager initialized")

    def retrieve(
        self,
        patient_id: str,
        user_query: str,
        short_term_context: str,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve long-term memories

        Args:
            patient_id: Patient ID
            user_query: User query
            short_term_context: Short-term memory context
            n_results: Number of results to return

        Returns:
            {
                "need_rag": bool,
                "rag_triggered": bool,
                "vector_results": List[Dict],
                "graph_results": List[Dict] | None,
                "formatted_context": str
            }
        """
        print(f"\nStarting long-term memory retrieval:")
        print(f"  Patient ID: {patient_id}")
        print(f"  Query: {user_query[:50]}...")

        # 1. RAG intent classification + query strategy determination
        rag_result = self.rag_classifier.classify_with_strategy(
            user_query=user_query,
            short_term_context=short_term_context
        )

        if not rag_result:
            print("  ✗ RAG intent classification failed")
            return self._empty_result()

        need_rag = rag_result['need_rag']
        print(f"  need_rag: {need_rag}")
        print(f"  reason: {rag_result['reason']}")

        if not need_rag:
            print("  → RAG not needed, short-term memory is sufficient")
            return self._empty_result()

        # 2. Get query strategy
        query_strategy = rag_result['query_strategy']
        print(f"  Query strategy: Vector DB={query_strategy['vector_db']}, Graph DB={query_strategy['graph_db']}")

        # 3. Vector database retrieval
        vector_results = []
        if query_strategy['vector_db']:
            vector_results = self._retrieve_from_vector_db(
                patient_id=patient_id,
                user_query=user_query,
                n_results=n_results
            )
            print(f"  ✓ Vector retrieval completed: Found {len(vector_results)} memories")

        # 4. Graph database retrieval (as needed)
        graph_results = None
        if query_strategy['graph_db']:
            graph_query_type = query_strategy['graph_query_type']
            graph_results = self._retrieve_from_graph_db(
                patient_id=patient_id,
                query_type=graph_query_type
            )
            print(f"  ✓ Graph retrieval completed: Found {len(graph_results) if graph_results else 0} records")

        # 5. Format results
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
        Retrieve from vector database

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
        # 1. Generate query vector
        query_embedding = self.sbert_model.encode(user_query).tolist()

        # 2. Query Chroma
        chroma_results = self.db.query_memory_by_vector(
            query_embedding=query_embedding,
            patient_id=patient_id,
            n_results=n_results
        )

        # 3. Parse results
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

            # Parse complete analysis results
            analysis = json.loads(metadata['analysis_json'])

            formatted_results.append({
                "unit_id": unit_id,
                "session_id": metadata['session_id'],
                "unit_type": metadata['unit_type'],
                "cluster_id": metadata.get('cluster_id'),
                "similarity": 1 - distance,  # Convert to similarity
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
        Retrieve from graph database

        Args:
            patient_id: Patient ID
            query_type: Query type (drug_interaction, symptom_disease, etc.)

        Returns:
            List of graph query results
        """
        try:
            results = self.db.query_graph(
                query_type=query_type,
                patient_id=patient_id
            )
            return results

        except Exception as e:
            print(f"  ⚠️  Graph query failed: {e}")
            return None

    def _format_retrieval_results(
        self,
        vector_results: List[Dict],
        graph_results: Optional[List[Dict]]
    ) -> str:
        """
        Format retrieval results as text context

        Used for providing context to LLM when generating responses

        Returns:
            Formatted context string
        """
        context_parts = []

        # 1. Vector retrieval results
        if vector_results:
            context_parts.append("【Historical Medical Records】\n")

            for i, result in enumerate(vector_results, 1):
                session_topic = result['session_topic']
                summary = result['narrative_summary']
                created_at = result['created_at'][:10]  # Only take the date part

                context_parts.append(
                    f"{i}. [{created_at}] {session_topic}\n"
                    f"   Summary: {summary}\n"
                )

        # 2. Graph retrieval results
        if graph_results:
            context_parts.append("\n【Related Medical Information】\n")

            # Format based on different query types
            if isinstance(graph_results, list) and graph_results:
                first_result = graph_results[0]

                # Drug interactions
                if 'drug1' in first_result and 'drug2' in first_result:
                    context_parts.append("Drug Interaction Warnings:\n")
                    for item in graph_results:
                        severity = item.get('severity', 'Unknown')
                        context_parts.append(
                            f"  - {item['drug1']} + {item['drug2']}: "
                            f"{severity} level interaction\n"
                        )

                # Symptom-disease associations
                elif 'symptom' in first_result and 'disease' in first_result:
                    context_parts.append("Possible Diseases Related to Symptoms:\n")
                    for item in graph_results:
                        context_parts.append(
                            f"  - Symptom '{item['symptom']}' may indicate '{item['disease']}'\n"
                        )

                # Treatment history
                elif 'drug' in first_result and 'prescribed_at' in first_result:
                    context_parts.append("Historical Medication Records:\n")
                    for item in graph_results:
                        date = item['prescribed_at'][:10]
                        dosage = item.get('dosage', 'Not recorded')
                        context_parts.append(
                            f"  - [{date}] {item['drug']} ({dosage})\n"
                        )

        return "".join(context_parts) if context_parts else ""

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        """Return empty result"""
        return {
            "need_rag": False,
            "rag_triggered": False,
            "vector_results": [],
            "graph_results": None,
            "formatted_context": ""
        }


# Global retrieval manager instance
_memory_retrieval: Optional[MemoryRetrieval] = None


def get_memory_retrieval() -> MemoryRetrieval:
    """Get global retrieval manager instance (singleton pattern)"""
    global _memory_retrieval
    if _memory_retrieval is None:
        _memory_retrieval = MemoryRetrieval()
    return _memory_retrieval


# Unit test
if __name__ == "__main__":
    print("=== Long-term Memory Retrieval Test ===\n")

    retrieval = MemoryRetrieval()

    # Test query
    print("【Test 1 - Query Requiring RAG】")
    result = retrieval.retrieve(
        patient_id="P12345",
        user_query="Can I continue taking the blood pressure medication prescribed by my doctor last time?",
        short_term_context="",
        n_results=3
    )

    print(f"\nRetrieval results:")
    print(f"  need_rag: {result['need_rag']}")
    if result['rag_triggered']:
        print(f"  Confidence: {result.get('confidence', 0):.2f}")
        print(f"  Reason: {result.get('reason', '')}")
        print(f"  Vector results count: {len(result['vector_results'])}")
        print(f"  Graph results: {'Yes' if result['graph_results'] else 'No'}")
        print(f"\nFormatted context:")
        print(result['formatted_context'][:200] if result['formatted_context'] else "(Empty)")

    print("\nTest completed")
