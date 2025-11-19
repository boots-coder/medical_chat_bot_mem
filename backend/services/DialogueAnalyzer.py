from typing import List, Dict
import json
from backend.ml.APIManager import APIManager


class TokenLimitExceeded(Exception):
    """Exception raised when dialogue token count exceeds limit"""
    pass


class DialogueAnalyzer:
    """
    Dialogue Analyzer: Converts medical dialogue information into three-tier storage structure,
    for subsequent storage; prompt engineering.
    Can be replaced with a fine-tuned small model in the future.
    """

    def __init__(
        self,
        api_manager: APIManager = None,
        max_input_tokens: int = 4096
    ):
        """
        Initialize dialogue analyzer

        Args:
            api_manager: API manager instance, creates default instance if None
            max_input_tokens: Maximum input token limit
        """
        self.api_manager = api_manager if api_manager else APIManager()
        self.max_input_tokens = max_input_tokens

        if not self.api_manager.is_available():
            print("⚠️ Warning: API unavailable, dialogue analysis feature will not work")

    def analyze_session(
        self,
        dialogue_list: List[Dict[str, str]],
        session_id: str,
        user_id: str,
        start_time: str,
        end_time: str
    ) -> Dict:
        """
        Analyze dialogue session, return three-tier storage structure

        Args:
            dialogue_list: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            session_id: Session ID
            user_id: User ID
            start_time: ISO8601 format start time
            end_time: ISO8601 format end time

        Returns:
            Dictionary containing SQLite, VectorDB, GraphDB data
        """
        # TODO: Implement token counting logic
        # estimated_tokens = self._estimate_tokens(dialogue_list)
        # if estimated_tokens > self.max_input_tokens:
        #     raise TokenLimitExceeded(f"Dialogue tokens({estimated_tokens}) exceed limit({self.max_input_tokens})")

        # Build prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(dialogue_list, session_id, user_id, start_time, end_time)

        # Use API manager to call LLM
        if not self.api_manager.is_available():
            raise RuntimeError("API unavailable, cannot perform dialogue analysis")

        result = self.api_manager.call_json_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,  # Low temperature ensures consistency of structured output
            max_tokens=3000  # Increase output length to accommodate complete knowledge graph
        )

        if not result:
            raise RuntimeError("Dialogue analysis API call failed")

        return result

    def _build_system_prompt(self) -> str:
        """Build system prompt"""
        return """You are a professional medical dialogue analysis assistant, responsible for converting doctor-patient dialogues into structured three-tier data storage format.

## Task Objective
Analyze medical dialogues, extract information and output JSON format, including three tiers:
1. **SQLite (Fact Layer)**: Session metadata (time, rounds, topic)
2. **VectorDB (Semantic Layer)**: Dialogue summary and semantic expression of chief complaint
3. **GraphDB (Relationship Layer)**: Knowledge graph of medical entities and their relationships

## Output Format Specification
```json
{
  "session_id": "Session ID (must come from user prompt)",
  "user_id": "User ID (must come from user prompt)",
  "start_time": "ISO8601 format start time (must come from user prompt)",
  "end_time": "ISO8601 format end time (must come from user prompt)",
  "dialogue_rounds": "Number of dialogue rounds (integer, must come from user prompt)",
  "session_topic": "One sentence summarizing dialogue topic",

  "narrative_summary": "3-5 sentences fully narrating dialogue process, including chief complaint, assessment, recommendations",
  "main_complaint_vectorized": "Core keyword combination extracted from user's chief complaint",

  "knowledge_graph": {
    "entities": [
      {"id": "Unique identifier", "type": "Entity type", "label": "Entity name"}
    ],
    "relationships": [
      {"subject": "Subject entity ID", "predicate": "Relationship type", "object": "Object entity ID"}
    ]
  }
}
```

## Entity Type Specification
- **Patient**: Patient (use user_id as ID)
- **Symptom**: Symptom (ID prefix: `S_`, e.g., `S_Headache`)
- **Disease**: Disease (ID prefix: `D_`, e.g., `D_Hypertension`)
- **Diagnosis**: Diagnostic conclusion (ID prefix: `DG_`, e.g., `DG_TensionHeadache`)
- **Drug**: Medication (ID prefix: `DR_`, e.g., `DR_Ibuprofen`)
- **Examination**: Examination item (ID prefix: `E_`, e.g., `E_BloodPressure`)
- **Treatment**: Treatment plan (ID prefix: `T_`, e.g., `T_BedRest`)

## Relationship Type Specification
- **HAS_SYMPTOM**: Patient has symptom
- **HAS_HISTORY**: Patient has medical history
- **MAY_CAUSE**: Disease may cause symptom
- **IS_SUGGESTED_FOR**: Diagnosis suggested for patient
- **RECOMMENDED_FOR**: Drug/treatment recommended for diagnosis
- **REQUIRES**: Diagnosis requires examination
- **INDICATES**: Examination result indicates disease

## Important Notes
1. Entity IDs use English, labels use original language
2. Timestamps use ISO 8601 format (UTC)
3. All entity IDs must be standardized (future integration with ICD-10, SNOMED CT and other medical coding libraries)
4. Relationships must accurately reflect medical logic
5. Must return valid JSON format
6. Referential integrity: Every `subject` and `object` `id` referenced in the 'relationships' list **must** be **declared in advance** in the `entities` list.

Please strictly follow the above format for output."""

    def _build_user_prompt(
        self,
        dialogue_list: List[Dict[str, str]],
        session_id: str,
        user_id: str,
        start_time: str,
        end_time: str
    ) -> str:
        """Build user prompt"""
        # Format dialogue history
        dialogue_text = ""
        for i, turn in enumerate(dialogue_list, 1):
            role = "User" if turn["role"] == "user" else "Assistant"
            dialogue_text += f"[Round {i} - {role}]: {turn['content']}\n\n"

        # Calculate actual number of rounds on Python side (one Q&A = 1 round)
        # Note: len(dialogue_list) is total turns, divide by 2 to get rounds
        dialogue_rounds = max(1, len(dialogue_list) // 2)  # Avoid 0

        return f"""Please analyze the following medical dialogue and extract structured information.

## Session Information (Please copy this information directly to JSON without modification)
- session_id: {session_id}
- user_id: {user_id}
- start_time: {start_time}
- end_time: {end_time}
- dialogue_rounds: {dialogue_rounds}

## Dialogue Content
{dialogue_text}

Please output analysis results according to the JSON format in the system prompt."""

    def _estimate_tokens(self, dialogue_list: List[Dict[str, str]]) -> int:
        """
        Estimate token count of dialogue

        TODO: Use tiktoken library for precise counting
        Simple implementation: 1 Chinese character ≈ 1.5 tokens, 1 English word ≈ 1.3 tokens
        """
        total_chars = sum(len(turn["content"]) for turn in dialogue_list)
        estimated_tokens = int(total_chars * 1.5)  # Rough estimate
        return estimated_tokens


# Usage example
if __name__ == "__main__":
    analyzer = DialogueAnalyzer(max_input_tokens=4096)

    # Simulate dialogue data
    dialogue = [
        {"role": "user", "content": "Hello, I've had a constant headache for the past two days, and I feel a bit nauseous"},
        {"role": "assistant", "content": "Hello, do you have hypertension or other chronic conditions?"},
        {"role": "user", "content": "Yes, I have hypertension for three years now"},
        {"role": "assistant", "content": "The headache may be related to elevated blood pressure. Recommend monitoring blood pressure, and if necessary, you can take ibuprofen to relieve symptoms"}
    ]

    try:
        result = analyzer.analyze_session(
            dialogue_list=dialogue,
            session_id="sess_1a2b3c",
            user_id="user_789",
            start_time="2025-11-04T12:30:00Z",
            end_time="2025-11-04T12:34:51Z"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (TokenLimitExceeded, RuntimeError) as e:
        print(f"Error: {e}")
        # TODO: Implement dialogue compression logic in the future
        # compressed_dialogue = compress_dialogue(dialogue)
        # result = analyzer.analyze_session(compressed_dialogue, ...)
