import json
from typing import Optional, Dict, Any
from backend.ml.APIManager import APIManager
from backend.models.ShortTermMemoryManager import ShortTermMemoryManager
from backend.ml.RAGIntentClassifier import RAGIntentClassifier


class MedicalResponseGenerator:
    """
    Medical Response Generator: Integrates three information sources to generate final response

    Information sources:
    1. User's current query
    2. Short-term memory (from ShortTermMemoryManager)
    3. Long-term memory (if RAGIntentClassifier determines RAG is needed)

    Integrates all components to generate personalized medical advice
    """

    def __init__(self, api_manager: APIManager = None):
        """
        Initialize medical response generator

        Args:
            api_manager: API manager instance, creates default instance if None
        """
        # Initialize API manager
        self.api_manager = api_manager if api_manager else APIManager()

        # Initialize components, sharing the same API manager
        self.memory_manager = ShortTermMemoryManager(max_tokens=2000, max_turns=8, api_manager=self.api_manager)
        self.rag_classifier = RAGIntentClassifier(api_manager=self.api_manager)

        if not self.api_manager.is_available():
            print("⚠️ Warning: API unavailable, medical response generation feature will not work")

    def _build_system_prompt(self) -> str:
        """Build system prompt for medical response generation"""
        return """You are a professional medical AI assistant with memory capabilities.

## Response Principles
1. **Concise and Precise**: Answer questions directly, avoid redundant information
2. **Safety First**: Recommend medical attention for serious symptoms, do not provide definitive diagnosis
3. **Personalized**: Use historical information to provide targeted advice
4. **Easy to Understand**: Avoid excessive medical terminology, patients should be able to understand

## Response Requirements
- Keep each response within 3-5 sentences
- Answer user's question directly, do not over-expand
- If there are historical records, brief comparison is sufficient
- Only give medication advice when necessary, do not proactively recommend drugs
- Avoid using formatted titles like "Symptom Analysis", "Personalized Advice", etc.

## Example Style
User: "What should I do about my headache?"
✅ Good response: "Based on your description, it may be tension headache. Recommend resting first and staying in a quiet environment. If pain worsens or persists beyond 24 hours, recommend seeing a doctor."
❌ Poor response: "**Symptom Analysis:** You have headache symptoms, which may be caused by multiple factors...**Personalized Advice:**...**Medication Guidance:**..."

Keep responses natural, concise, and targeted."""

    def generate_response(
        self,
        user_query: str,
        short_term_context: str = "",
        long_term_memory: str = ""
    ) -> Dict[str, Any]:
        """
        Generate medical response

        Args:
            user_query: User's current query
            short_term_context: Short-term memory context (obtained from SessionManager)
            long_term_memory: Long-term memory context (retrieved by MemoryRetrieval)

        Returns:
            {
                "response": str,           # Generated response
                "used_short_memory": str,  # Short-term memory used
                "used_long_memory": str,   # Long-term memory used
                "rag_triggered": bool,     # Whether RAG was triggered
                "confidence": float        # RAG classification confidence
            }
        """
        if not self.api_manager.is_available():
            return {
                "response": "Error: API unavailable",
                "used_short_memory": "",
                "used_long_memory": "",
                "rag_triggered": False,
                "confidence": 0.0
            }

        # 1. Use passed short-term memory context
        short_memory_context = short_term_context

        # 2. RAG intent classification
        rag_result = self.rag_classifier.classify_rag_intent(user_query, short_memory_context)
        rag_triggered = rag_result["need_rag"] if rag_result else False
        confidence = rag_result["confidence"] if rag_result else 0.0

        # 3. Use passed long-term memory content (already retrieved by external RAG)
        final_long_memory = long_term_memory if long_term_memory else ""

        # 4. Build complete context prompt
        context_parts = []

        if short_memory_context:
            context_parts.append(f"Short-term memory context:\n{short_memory_context}")

        if final_long_memory:
            context_parts.append(f"Long-term memory context:\n{final_long_memory}")

        context_parts.append(f"User's current query: {user_query}")

        full_context = "\n\n".join(context_parts)

        # 5. Call LLM to generate response
        response = self.api_manager.call_text_completion(
            system_prompt=self._build_system_prompt(),
            user_prompt=full_context,
            temperature=0.3,  # Slightly higher to maintain response naturalness
            max_tokens=800
        )

        if response:
            return {
                "response": response,
                "used_short_memory": short_memory_context,
                "used_long_memory": final_long_memory,
                "rag_triggered": rag_triggered,
                "confidence": confidence
            }
        else:
            return {
                "response": "Sorry, system temporarily cannot generate response",
                "used_short_memory": short_memory_context,
                "used_long_memory": final_long_memory,
                "rag_triggered": rag_triggered,
                "confidence": confidence
            }

    def add_conversation_turn(self, role: str, content: str):
        """Add conversation turn to short-term memory"""
        self.memory_manager.add_turn(role, content)

    def new_session(self):
        """Start new session (clear short-term memory)"""
        self.memory_manager.clear()
        print("[System] New session started, short-term memory cleared")

    def get_memory_stats(self) -> Dict:
        """Get memory status statistics"""
        return self.memory_manager.get_stats()
