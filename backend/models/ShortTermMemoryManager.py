import json
from typing import List, Dict, Optional
from backend.ml.APIManager import APIManager


class ShortTermMemoryManager:
    """
    Short-term Memory Manager: Implements buffer functionality similar to LangChain

    Features:
    1. Maintain current session's dialogue window
    2. Summarize when tokens exceed limit
    3. Keep historical summary + current window
    4. Provide short-term memory context for RAG intent classification and final response
    """

    def __init__(self, max_tokens: int = 2000, max_turns: int = 10, api_manager: APIManager = None):
        """
        Args:
            max_tokens: Maximum token count, triggers summary when exceeded
            max_turns: Maximum dialogue rounds, triggers summary when exceeded
            api_manager: API manager instance, creates default instance if None
        """
        # Memory management parameters
        self.max_tokens = max_tokens
        self.max_turns = max_turns

        # Memory state
        self.historical_summary = ""  # Historical summary
        self.current_window = []      # Current window dialogue
        self.estimated_tokens = 0     # Estimated token count

        # Initialize API manager
        self.api_manager = api_manager if api_manager else APIManager()
        if not self.api_manager.is_available():
            print("⚠️ Warning: API unavailable, short-term memory summary feature will not work")

    def _estimate_tokens(self, text: str) -> int:
        """
        Roughly estimate token count of text (Chinese characters * 1.5)
        TODO: Can use tiktoken for precise calculation in the future
        """
        return int(len(text) * 1.5)

    def _build_summarization_prompt(self) -> str:
        """Build summarization prompt"""
        return """You are a medical dialogue summary assistant, need to summarize multi-round doctor-patient dialogues into concise context summaries.

## Summarization Principles
1. Preserve all medically relevant key information: symptoms, medical history, medications, diagnostic recommendations
2. Maintain temporal order and causal relationships
3. Remove irrelevant pleasantries and redundant information
4. Keep summary length within 200 characters

## Output Format
Return JSON format:
```json
{
  "summary": "Concise dialogue summary including key medical information",
  "key_points": ["Point 1", "Point 2", "Point 3"]
}
```

Requirements:
- summary: Coherent narrative summary
- key_points: 3-5 key points list"""

    def _summarize_conversations(self) -> Optional[str]:
        """
        Summarize current window's dialogue

        Returns:
            Summarized text, returns None on failure
        """
        if not self.api_manager.is_available() or not self.current_window:
            return None

        # Build dialogue text
        dialogue_text = ""
        for i, turn in enumerate(self.current_window, 1):
            role = "User" if turn["role"] == "user" else "Assistant"
            dialogue_text += f"[Round {i} - {role}]: {turn['content']}\n\n"

        # If there's historical summary, include it too
        context_text = ""
        if self.historical_summary:
            context_text = f"Historical summary: {self.historical_summary}\n\nCurrent dialogue:\n{dialogue_text}"
        else:
            context_text = f"Dialogue content:\n{dialogue_text}"

        # Use API manager for the call
        result = self.api_manager.call_json_completion(
            system_prompt=self._build_summarization_prompt(),
            user_prompt=context_text,
            temperature=0.1,
            max_tokens=300
        )

        if result:
            return result.get("summary", "")
        else:
            print("Dialogue summarization failed")
            return None

    def add_turn(self, role: str, content: str) -> bool:
        """
        Add new dialogue turn

        Args:
            role: "user" or "assistant"
            content: Dialogue content

        Returns:
            Whether summary was triggered
        """
        # Add to current window
        turn = {"role": role, "content": content}
        self.current_window.append(turn)

        # Update token estimate
        self.estimated_tokens += self._estimate_tokens(content)

        # Check if summary is needed
        triggered_summary = False
        if (self.estimated_tokens > self.max_tokens or
            len(self.current_window) >= self.max_turns * 2):  # *2 because one round = user + assistant

            # Perform summary
            new_summary = self._summarize_conversations()
            if new_summary:
                self.historical_summary = new_summary
                self.current_window = []  # Clear current window
                self.estimated_tokens = self._estimate_tokens(self.historical_summary)
                triggered_summary = True
                print(f"[Short-term Memory] Summary triggered, new summary length: {len(self.historical_summary)} characters")

        return triggered_summary

    def get_context(self) -> str:
        """
        Get complete short-term memory context

        Returns:
            Complete context including historical summary + current window
        """
        context_parts = []

        # Add historical summary
        if self.historical_summary:
            context_parts.append(f"Historical summary: {self.historical_summary}")

        # Add current window
        if self.current_window:
            current_text = ""
            for turn in self.current_window:
                role = "User" if turn["role"] == "user" else "Assistant"
                current_text += f"{role}: {turn['content']}\n"
            context_parts.append(f"Current dialogue:\n{current_text.strip()}")

        return "\n\n".join(context_parts) if context_parts else ""

    def get_stats(self) -> Dict:
        """Get memory status statistics"""
        return {
            "historical_summary_length": len(self.historical_summary),
            "current_window_rounds": len(self.current_window),
            "estimated_tokens": self.estimated_tokens,
            "has_historical_summary": bool(self.historical_summary)
        }

    def clear(self):
        """Clear all memory (called when new session starts)"""
        self.historical_summary = ""
        self.current_window = []
        self.estimated_tokens = 0


# Unit tests
def test_short_term_memory():
    """Short-term memory manager unit test"""
    print("="*60)
    print("Short-term Memory Manager Unit Test")
    print("="*60)

    # Create instance with smaller limits for testing convenience
    memory = ShortTermMemoryManager(max_tokens=500, max_turns=3)

    print("\n【Test Scenario】Simulate a gradually growing medical dialogue")
    print("-" * 40)

    # Simulate conversation sequence
    conversation = [
        ("user", "I've been having severe headaches lately, especially in the temple area"),
        ("assistant", "When did your headache start? Is it continuous or intermittent?"),
        ("user", "Started about three days ago, comes in waves, especially in the afternoon"),
        ("assistant", "Sounds like it might be migraine. Have you had similar situations before? Any nausea or vomiting?"),
        ("user", "I've had it before, but not this severe. A bit nauseous, but no vomiting"),
        ("assistant", "Recommend you take ibuprofen to relieve symptoms, also rest and avoid strong light stimulation"),
        ("user", "How should I take ibuprofen? How many times a day?"),
        ("assistant", "Generally 400mg each time, 2-3 times a day, after meals. If symptoms persist, recommend seeing a doctor"),
        ("user", "Okay, thank you doctor. By the way, could it be related to my recent work stress?"),
        ("assistant", "Quite possible. Stress, lack of sleep, irregular diet can all trigger migraines. Recommend adjusting your schedule")
    ]

    # Gradually add dialogue
    for i, (role, content) in enumerate(conversation, 1):
        print(f"\n[Round {i}] {role}: {content[:50]}...")

        # Add dialogue
        triggered = memory.add_turn(role, content)

        # Show status
        stats = memory.get_stats()
        print(f"   Status: window={stats['current_window_rounds']}, tokens≈{stats['estimated_tokens']}, summary={'Yes' if triggered else 'No'}")

        # If summary was triggered, show summary
        if triggered:
            print(f"   📋 Generated summary: {memory.historical_summary[:80]}...")

    print(f"\n【Final Short-term Memory Context】")
    print("-" * 40)
    final_context = memory.get_context()
    print(final_context[:300] + "..." if len(final_context) > 300 else final_context)

    print(f"\n【Final Statistics】")
    print(memory.get_stats())

    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)


if __name__ == "__main__":
    test_short_term_memory()
