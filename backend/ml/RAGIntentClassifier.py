import json
from typing import Optional
from backend.ml.APIManager import APIManager


class RAGIntentClassifier:
    """
    RAG Intent Classifier: Determines if user query requires long-term memory retrieval and query strategy

    Features:
    1. Determine if RAG retrieval is needed (need_rag: bool)
    2. If RAG is needed, further determine query strategy:
       - Whether graph database query is needed (graph_db: bool)
       - Graph database query type (graph_query_type: str)

    Integrates with ShortTermMemoryManager, based on current query and short-term memory context
    determines if cross-session RAG query needs to be triggered

    Prerequisite: User visit count > 1 (has historical records)
    """

    # Graph query type definitions
    GRAPH_QUERY_TYPES = {
        "drug_interaction": {
            "keywords": ["medication", "drug", "together", "conflict", "interaction", "simultaneously", "affect"],
            "description": "Drug interaction query"
        },
        "symptom_disease": {
            "keywords": ["symptom", "what disease", "possibly", "diagnosis", "what cause"],
            "description": "Symptom-disease association query"
        },
        "diagnosis_chain": {
            "keywords": ["medical history", "history", "before", "previously", "diagnosed", "treated", "record"],
            "description": "Diagnosis chain query"
        },
        "treatment_history": {
            "keywords": ["treatment plan", "effect", "followup", "efficacy", "improvement"],
            "description": "Treatment history query"
        }
    }

    def __init__(self, api_manager: APIManager = None):
        """
        Initialize RAG intent classifier

        Args:
            api_manager: API manager instance, creates default instance if None
        """
        # Initialize API manager
        self.api_manager = api_manager if api_manager else APIManager()
        if not self.api_manager.is_available():
            print("⚠️ Warning: API unavailable, RAG intent classification feature will not work")

    def _build_system_prompt(self) -> str:
        """Build RAG intent classification system prompt"""
        return """You are a professional RAG intent classifier, specially designed for medical dialogue systems.

Your task is to determine if the user's query requires retrieval of **long-term memory** (cross-session historical records).

## Background
- Short-term memory: Current session's dialogue context (including historical summary + current window), always available
- Long-term memory: Cross-session historical medical records, requires RAG retrieval, higher cost
- System will provide: User's current query + complete short-term memory context

## Classification Principles

### Requires RAG Retrieval (needs long-term memory) - return true
1. **Explicit historical references**:
   - "What the doctor said last time...", "Previous examination results..."
   - "I had this before...", "In my medical history..."
   - "When I came to see the doctor last time..."

2. **Symptom comparison and trends**:
   - "Compared to last time's headache..."
   - "More severe/better than before"
   - "Same problem again", "Old problem recurred"

3. **Treatment effect tracking**:
   - "After taking the medicine prescribed last time..."
   - "Following previous advice..."
   - "Effect of last treatment plan..."

4. **Chronic disease management inquiries**:
   - "How is my hypertension controlled"
   - "Recent changes in diabetes"
   - "Regular checkup result comparison"

### Does Not Require RAG Retrieval (short-term memory sufficient) - return false
1. **New symptom description**:
   - First mention of symptom or feeling
   - Current physical condition description

2. **General medical consultation**:
   - General disease knowledge inquiry
   - Medication usage methods

3. **Current dialogue continuation**:
   - Further inquiry about just-mentioned content
   - Short-term memory context already contains sufficient information

## Key Decision Points
- If user uses time words like "last time", "before", "previously", and **not** referring to current session content → needs RAG
- If short-term memory context already contains information related to user's inquiry → no need for RAG
- If it's historical comparison or trend analysis of symptoms → needs RAG

## Output Format
You must only return a JSON object:
```json
{
  "need_rag": true/false,
  "confidence": 0.0-1.0,
  "reason": "Brief explanation of decision basis"
}
```

Note:
- need_rag: true means long-term memory retrieval needed, false means short-term memory sufficient
- confidence: Confidence score (0.0-1.0)
- reason: One sentence explaining the core basis of the decision"""

    def classify_rag_intent(self, user_query: str, short_term_context: str) -> Optional[dict]:
        """
        Classify RAG intent of user query

        Args:
            user_query: User's current query content
            short_term_context: Context provided by short-term memory manager

        Returns:
            {
                "need_rag": bool,        # Whether RAG retrieval is needed
                "confidence": float,     # Confidence 0.0-1.0
                "reason": str           # Reason for decision
            }
            Returns None if API call fails
        """
        if not self.api_manager.is_available():
            print(f"Error: API unavailable - unable to classify: '{user_query}'")
            return None

        # Build user prompt
        if short_term_context.strip():
            user_prompt = f"""User's current query: {user_query}

Short-term memory context:
{short_term_context}

Please determine based on the user's query and short-term memory context whether long-term memory retrieval is needed."""
        else:
            user_prompt = f"""User's current query: {user_query}

Short-term memory context: (empty, new session starting)

Please determine if long-term memory retrieval is needed."""

        # Use API manager for the call
        result = self.api_manager.call_json_completion(
            system_prompt=self._build_system_prompt(),
            user_prompt=user_prompt,
            temperature=0.1,  # Ensure classification consistency
            max_tokens=200
        )

        if not result:
            print(f"RAG intent classification API call failed")
            return None

        # Validate return format
        required_keys = ["need_rag", "confidence", "reason"]
        if not all(key in result for key in required_keys):
            print(f"Warning: API return format incomplete: {result}")
            return None

        # Validate data types
        if not isinstance(result["need_rag"], bool):
            print(f"Warning: need_rag is not boolean: {result['need_rag']}")
            return None

        if not isinstance(result["confidence"], (int, float)) or not (0 <= result["confidence"] <= 1):
            print(f"Warning: confidence not in 0-1 range: {result['confidence']}")
            return None

        return result

    def quick_check(self, user_query: str, short_term_context: str = "") -> bool:
        """
        Quick check if RAG is needed (simplified version, only returns True/False)

        Args:
            user_query: User query
            short_term_context: Short-term memory context

        Returns:
            True: RAG retrieval needed, False: No need for RAG retrieval or API call failed
        """
        result = self.classify_rag_intent(user_query, short_term_context)
        return result["need_rag"] if result else False

    def classify_query_strategy(self, user_query: str) -> dict:
        """
        Rule-based query strategy classification (no LLM call, fast decision)

        Determine if graph database query is needed and what type of graph query to use

        Args:
            user_query: User query content

        Returns:
            {
                "vector_db": True,           # Always use vector database
                "graph_db": bool,            # Whether graph database is needed
                "graph_query_type": str|None # Graph query type
            }
        """
        query_lower = user_query.lower()

        # Default strategy: only use vector retrieval
        strategy = {
            "vector_db": True,
            "graph_db": False,
            "graph_query_type": None
        }

        # Check if matches graph query pattern
        for query_type, config in self.GRAPH_QUERY_TYPES.items():
            if any(keyword in query_lower for keyword in config["keywords"]):
                strategy["graph_db"] = True
                strategy["graph_query_type"] = query_type
                break  # Stop at first match

        return strategy

    def classify_with_strategy(self, user_query: str, short_term_context: str) -> Optional[dict]:
        """
        Complete RAG intent classification + query strategy determination

        Args:
            user_query: User's current query content
            short_term_context: Context provided by short-term memory manager

        Returns:
            {
                "need_rag": bool,
                "confidence": float,
                "reason": str,
                "query_strategy": {
                    "vector_db": bool,
                    "graph_db": bool,
                    "graph_query_type": str|None
                }
            }
            Returns None if API call fails
        """
        # Step 1: RAG intent classification (LLM call)
        rag_result = self.classify_rag_intent(user_query, short_term_context)

        if not rag_result:
            return None

        # Step 2: If RAG is needed, perform query strategy classification (rule-based)
        if rag_result["need_rag"]:
            strategy = self.classify_query_strategy(user_query)
        else:
            # No need for RAG, strategy is meaningless
            strategy = {
                "vector_db": False,
                "graph_db": False,
                "graph_query_type": None
            }

        # Merge results
        rag_result["query_strategy"] = strategy
        return rag_result


# Unit tests
def test_rag_intent_classifier():
    """RAG intent classifier unit test (integrated with short-term memory manager)"""
    print("="*60)
    print("RAG Intent Classifier Unit Test")
    print("="*60)

    classifier = RAGIntentClassifier()

    # Test case 1: Requires RAG - historical symptom comparison
    print("\n【Test Case 1 - Requires RAG】")
    print("Scenario: User mentions historical comparison while discussing current headache")
    print("-" * 50)

    short_context_1 = """Historical summary: User complained of headache for three days, intermittent pain in temple area, with nausea. Possible diagnosis of migraine, recommended ibuprofen.

Current dialogue:
User: I took ibuprofen as you suggested, feeling better
Assistant: Good, symptom relief is a good sign. Please continue to observe"""

    query_1 = "Is this headache more severe compared to when I came last time?"

    result_1 = classifier.classify_rag_intent(query_1, short_context_1)
    if result_1:
        status = "✓" if result_1["need_rag"] else "✗"
        print(f"{status} Query: {query_1}")
        print(f"   Result: need_rag={result_1['need_rag']}, confidence={result_1['confidence']:.2f}")
        print(f"   Reason: {result_1['reason']}")
        print(f"   Context: {short_context_1[:100]}...")

    # Test case 2: Does not require RAG - current session content sufficient
    print("\n【Test Case 2 - Does Not Require RAG】")
    print("Scenario: User's inquiry is already contained in short-term memory")
    print("-" * 50)

    short_context_2 = """Current dialogue:
User: I have stomach pain, want to take some medicine
Assistant: I recommend you take omeprazole, half an hour before meals, once a day
User: Okay, thank you doctor"""

    query_2 = "How long should I take the omeprazole you just mentioned?"

    result_2 = classifier.classify_rag_intent(query_2, short_context_2)
    if result_2:
        status = "✓" if not result_2["need_rag"] else "✗"
        print(f"{status} Query: {query_2}")
        print(f"   Result: need_rag={result_2['need_rag']}, confidence={result_2['confidence']:.2f}")
        print(f"   Reason: {result_2['reason']}")
        print(f"   Context: {short_context_2}")

    # Test case 3: Edge case - empty context but mentions history
    print("\n【Test Case 3 - Edge Case: Empty Context But Mentions History】")
    print("-" * 50)

    query_3 = "Can I continue taking the hypertension medicine the doctor prescribed last time?"
    short_context_3 = ""

    result_3 = classifier.classify_rag_intent(query_3, short_context_3)
    if result_3:
        status = "✓" if result_3["need_rag"] else "✗"
        print(f"{status} Query: {query_3}")
        print(f"   Result: need_rag={result_3['need_rag']}, confidence={result_3['confidence']:.2f}")
        print(f"   Reason: {result_3['reason']}")

    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)


def test_query_strategy():
    """Test query strategy classification feature"""
    print("="*60)
    print("Query Strategy Classification Test")
    print("="*60)

    classifier = RAGIntentClassifier()

    test_cases = [
        ("Will my current blood pressure medication conflict with pain medication?", "drug_interaction"),
        ("What disease might these symptoms indicate?", "symptom_disease"),
        ("My diabetes treatment history records", "diagnosis_chain"),
        ("How was the effect of last treatment plan?", "treatment_history"),
        ("What medicine should I take for my headache?", None),  # No graph query needed
    ]

    for query, expected_type in test_cases:
        strategy = classifier.classify_query_strategy(query)
        print(f"\nQuery: {query}")
        print(f"  Vector DB: {strategy['vector_db']}")
        print(f"  Graph DB: {strategy['graph_db']}")
        print(f"  Graph Query Type: {strategy['graph_query_type']}")

        if expected_type:
            status = "✓" if strategy['graph_query_type'] == expected_type else "✗"
            print(f"  {status} Expected Type: {expected_type}")

    print("\n" + "="*60)
    print("Strategy Test Complete")
    print("="*60)


def test_classify_with_strategy():
    """Test complete classification flow (intent + strategy)"""
    print("="*60)
    print("Complete Classification Flow Test (Intent + Strategy)")
    print("="*60)

    classifier = RAGIntentClassifier()

    # Test scenario: Requires RAG and requires graph query
    short_context = """Historical summary: Patient has history of hypertension, currently taking antihypertensive medication.

Current dialogue:
User: I've been having severe headaches lately
Assistant: Your headache may be related to blood pressure, recommend monitoring blood pressure"""

    query = "Can I take ibuprofen together with my current blood pressure medication?"

    print(f"\nQuery: {query}")
    print(f"Short-term context: {short_context[:50]}...")

    result = classifier.classify_with_strategy(query, short_context)

    if result:
        print(f"\n【RAG Intent】")
        print(f"  need_rag: {result['need_rag']}")
        print(f"  confidence: {result['confidence']:.2f}")
        print(f"  reason: {result['reason']}")

        print(f"\n【Query Strategy】")
        print(f"  Vector DB: {result['query_strategy']['vector_db']}")
        print(f"  Graph DB: {result['query_strategy']['graph_db']}")
        print(f"  Graph Query Type: {result['query_strategy']['graph_query_type']}")

    print("\n" + "="*60)
    print("Complete Test Finished")
    print("="*60)


if __name__ == "__main__":
    # test_rag_intent_classifier()
    # test_query_strategy()
    test_classify_with_strategy()
