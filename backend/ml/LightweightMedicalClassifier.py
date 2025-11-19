import json
from typing import Dict, List, Tuple, Optional
from backend.ml.APIManager import APIManager

class LightweightMedicalClassifier:
    """
    A lightweight medical relevance classifier (secondary gatekeeper).
    Used to filter out context information corresponding to clustered centroids.
    """

    def __init__(self, api_manager: APIManager = None):
        """
        Initialize lightweight medical classifier

        Args:
            api_manager: API manager instance, creates default instance if None
        """
        # TODO: Replace this with a local SFT (Supervised Fine-Tuning) small model in the future

        # Initialize API manager
        self.api_manager = api_manager if api_manager else APIManager()
        if not self.api_manager.is_available():
            print("⚠️ Warning: API unavailable, medical classification feature will not work")



    def _build_system_prompt(self) -> str:
        """
        Build the classifier's system prompt.
        Optimized for the "James problem", clearly distinguishing "patient medical records" from "third-party news".
        """
        return """You are an efficient, lightweight medical relevance binary classifier.

Your sole task is to determine whether the input text is **directly related** to the **user's own (patient's)** medical, health, disease, symptom, or physical injury.

## Core Principles
We only care about the user's own (first person "I") or their consultation subject's (e.g., "my mom") health condition. Our goal is to filter out "valuable medical record information" for the doctor.

## Classification Rules

### Relevant (is_medical: true)
- User's own symptoms (e.g., headache, nausea, can't sleep, anxiety)
- User's own diseases (e.g., hypertension, diabetes)
- User's own medications (e.g., ibuprofen)
- User's own physical injuries (e.g., sprained ankle, fracture, cut)
- Family-related medical consultations (e.g., "My mom has hypertension...")
- Hypothetical medical consultations (e.g., "If I take cephalosporin...")

### Not Relevant (is_medical: false)
- Pure social greetings (e.g., hello, thank you, goodbye)
- Pure chitchat (e.g., nice weather, are you a robot)
- APP technical issues (e.g., network lag, how to use)
- Third-party/celebrity news: Medical events involving third parties (such as celebrities, news) unrelated to the user's own medical records.
- Pure metaphor: Clear exaggeration or metaphor (e.g., "I'm so angry my blood pressure is high")

## Key Boundary Examples
- "I like watching basketball" -> `{"is_medical": false}`
- "I got injured playing basketball" -> `{"is_medical": true}`
- "My mom has hypertension" -> `{"is_medical": true}` (family consultation, relevant)
- "Watching basketball makes my heart race" -> `{"is_medical": false}` (metaphor, not relevant)
- "I saw news that LeBron James sprained his ankle" -> `{"is_medical": false}` (third-party news, not related to patient's medical record)
- "My boss makes me so angry my blood pressure rises" -> `{"is_medical": false}` (exaggerated metaphor, not relevant)
- "I feel like my life is sick" -> `{"is_medical": false}` (pure philosophical metaphor)
- "I feel like my life is sick, I have no energy for anything lately" -> `{"is_medical": true}` (the latter part is a "fatigue" symptom)

## Output Format
You must only return a JSON object in this format:
`{"is_medical": <true_or_false>}`
Do not return any other text, pleasantries, or explanations.
"""
    def classify(self, text: str) -> Optional[bool]:
        """
        Classify a single sentence.

        Args:
            text: Sentence to classify.

        Returns:
            - True: Medical-related
            - False: Non-medical
            - None: API call or parsing failed
        """
        if not self.api_manager.is_available():
            print(f"Error: API unavailable - unable to classify: '{text}'")
            return None

        # Use API manager for the call
        user_prompt = f"Please classify the following text:\n\n{text}"

        result = self.api_manager.call_json_completion(
            system_prompt=self._build_system_prompt(),
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=50
        )

        if not result:
            print(f"API call or JSON parsing failed")
            return None

        is_medical = result.get("is_medical")  # No default value

        if isinstance(is_medical, bool):
            return is_medical
        else:
            print(f"Warning: API returned unexpected non-boolean value: {is_medical}")
            return None  # Treat as parsing failure


