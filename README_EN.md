# **Medical Chat Memory Manager**

An intelligent medical chatbot dialogue memory management system with short-term memory, long-term memory, and RAG (Retrieval-Augmented Generation) capabilities.

## Project Overview

**Scenario**: A patient receives a URL after booking an appointment, clicks it to access initial consultation, receives personalized Q&A (if they've visited before), and helps doctors quickly understand the patient's condition during the consultation.

**System Design Philosophy**: The system does not allow user access after the session ends (security & privacy).

## Core Features

- **Short-term Memory**: Maintains context within a single session using sliding window and summarization
- **Long-term Memory**: Cross-session historical medical records stored in multiple databases
- **RAG Intent Classification**: Intelligently determines when to retrieve historical data
- **Three-tier Storage Architecture**:
  - **SQLite**: Fact layer (session metadata)
  - **Chroma Vector DB**: Semantic layer (dialogue summaries)
  - **Neo4j Graph DB**: Relationship layer (medical knowledge graph)
- **Context-aware Clustering**: Handles ultra-long dialogue sessions
- **Medical Relevance Classification**: Filters non-medical content

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (Chat UI)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │    FastAPI Backend (WebSocket)   │
        └────────────────┬────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │       Session Manager            │
        │  (Token, Lifecycle, STM)         │
        └────────────────┬────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │  Medical Response Generator      │
        └────────────────┬────────────────┘
                         │
        ┌────────────────┴────────────────────┐
        │        Memory System                 │
        ├──────────────────┬──────────────────┤
        │  Short-term      │    Long-term     │
        │  Memory Manager  │  Memory (RAG)    │
        └──────────────────┴──────────────────┘
                         │
        ┌────────────────┴────────────────────┐
        │        Storage Layer                 │
        ├─────────────┬──────────┬────────────┤
        │   SQLite    │  Chroma  │   Neo4j    │
        │  (Metadata) │ (Vector) │  (Graph)   │
        └─────────────┴──────────┴────────────┘
```

## Project Structure

```
medical_chat_memory_manager/
├── backend/
│   ├── api/
│   │   └── main.py              # FastAPI main application
│   ├── core/
│   │   ├── config.py            # Configuration management
│   │   ├── database_schemas.py  # Database schema definitions
│   │   └── DatabaseManager.py   # Unified database manager
│   ├── ml/
│   │   ├── APIManager.py                      # LLM API manager
│   │   ├── LightweightMedicalClassifier.py   # Medical relevance classifier
│   │   ├── RAGIntentClassifier.py            # RAG intent classifier
│   │   └── context_aware_clusterer.py        # Context-aware dialogue clusterer
│   ├── models/
│   │   └── ShortTermMemoryManager.py  # Short-term memory manager
│   └── services/
│       ├── DialogueAnalyzer.py        # Dialogue analyzer
│       ├── MemoryStorage.py           # Long-term memory storage
│       ├── MemoryRetrieval.py         # Long-term memory retrieval
│       ├── SessionManager.py          # Session manager
│       ├── TokenManager.py            # Token & JWT manager
│       └── MedicalResponseGenerator.py # Medical response generator
├── frontend/                   # Chat interface (HTML/JS)
├── data/                      # Database files
├── init_db.py                 # Database initialization script
├── run.py                     # Application startup script
├── check_memory.py            # Chroma database query tool
└── requirements.txt           # Python dependencies
```

## Installation & Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create `.env` file:

```env
# LLM API Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4

# Server Configuration
PORT=8000
HOST=0.0.0.0

# Database Paths
SQLITE_DB_PATH=./data/medical_sessions.db
CHROMA_PERSIST_PATH=./data/chroma_db
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Memory Configuration
SHORT_TERM_MAX_TOKENS=2000
SHORT_TERM_MAX_TURNS=8
MAX_DIALOGUE_TURNS=20  # Threshold for triggering clustering

# Security
JWT_SECRET_KEY=your_secret_key_here
TOKEN_EXPIRY_HOURS=24
SESSION_TIMEOUT_MINUTES=30
```

### 3. Initialize Databases

```bash
# Initialize SQLite and Chroma
python init_db.py

# Start Neo4j (Docker) which is oprtional 
bash start_neo4j.sh
```

### 4. Run Application

```bash
python run.py
```

Access at: `http://localhost:8000`

## Key Concepts

### Session

A "session" is like a complete clinic visit:
- **Session Start**: Patient enters and says "Doctor, I don't feel well"
- **Session In Progress**: Doctor asks questions, patient answers (multi-turn dialogue)
- **Session End**:
  - **Explicit End**: Patient says "thank you, goodbye"
  - **Timeout End**: No message for 30 minutes

After session ends, dialogue is summarized and stored as long-term memory.

### Short-term Memory

**Current session's context information:**
- Semantic coherence
- Dialogue turns + generated summaries for semantic transfer
- Implemented using sliding window approach
- Inevitable semantic loss encourages users to express clearly in each turn

### Long-term Memory

**Information for cross-session queries:**
- Stored in different database structures
- Minimum storage unit depends on session dialogue length:
  1. Single session (short dialogues)
  2. Cluster (long dialogues)

### Three-tier Storage

**1. SQLite (Fact Layer)**
- Purpose: Exact lookups
- Stores: Facts like `user_id`, `session_id`, `start_time`
- Query: "Find all historical session IDs for Zhang San"

**2. Chroma Vector DB (Semantic Layer)**
- Purpose: Fuzzy matching / similarity search
- Stores: Experience and descriptions (dialogue summaries)
- Query: "Find all consultations similar to 'headache and nausea'"

**3. Neo4j Graph DB (Relationship Layer)**
- Purpose: Complex relationship queries
- Stores: Medical entities and their relationships
- Entities: Patient, Symptom, Disease, Diagnosis, Drug, Examination, Treatment
- Relationships: HAS_SYMPTOM, HAS_HISTORY, MAY_CAUSE, IS_SUGGESTED_FOR, RECOMMENDED_FOR

## Core Workflow

### 1. Session Creation

```
Patient books appointment → System generates URL with JWT token →
Patient receives URL → Click to access → Token validation →
Session created → Short-term memory initialized
```

### 2. Dialogue Process

```
User sends message →
Add to short-term memory →
RAG intent classification (need historical data?) →
If yes: Retrieve from long-term memory →
Medical Response Generator (STM + LTM + Query) →
Generate response → Send to user
```

### 3. Session End & Storage

```
Session ends (explicit/timeout) →
Check dialogue length →
If short: Direct storage →
If long: Clustering → Medical relevance filter →
DialogueAnalyzer extracts structured info →
Store to Vector DB + Graph DB
```

## Advanced Features

### Context-aware Clustering

For ultra-long dialogues (>20 rounds):
1. Convert each Q&A pair to embedding with historical context
2. Use DBSCAN for clustering
3. Filter each cluster for medical relevance
4. Store each cluster as independent unit

**Benefits**:
- Preserves focused medical information
- Reduces noise from off-topic conversations
- Enables fine-grained retrieval

### Medical Relevance Classification

Lightweight classifier filters out:
- Social greetings
- Chitchat
- App technical issues
- Third-party news/celebrity medical events
- Pure metaphors

Keeps only:
- User's own symptoms/diseases
- Family medical consultations
- Medical advice requests

## API Endpoints

### WebSocket Endpoint

```
WS /chat/{url_token}
```

**Message Format**:
```json
{
  "type": "user_message",
  "content": "I have a headache"
}
```

**Response Format**:
```json
{
  "type": "assistant_message",
  "content": "When did the headache start?...",
  "rag_triggered": false,
  "memory_stats": {...}
}
```

### REST API

```
POST /api/sessions/create
GET /api/sessions/{session_id}/stats
POST /api/sessions/{session_id}/end
```

## Development & Testing

### Run Unit Tests

```bash
# Test individual components
python -m backend.ml.RAGIntentClassifier
python -m backend.models.ShortTermMemoryManager
python -m backend.services.DialogueAnalyzer
```

### Check Vector Database

```bash
python check_memory.py
```

### Database Schema

See [backend/core/database_schemas.py](backend/core/database_schemas.py) for complete schema definitions.

## Why Not Use Mem0?

We built a custom framework instead of using Mem0 because:
1. **Different storage timing & data granularity**: Mem0 adds every sentence blindly; we store after session ends with intelligent filtering
2. **Complete data control**: Full ownership of storage logic and data structure
3. **Absolute control over information extraction**: Easy to standardize graph DB schema for medical domain
4. **Customization**: Our architecture is a medical-domain-specific implementation of Mem0's multi-tier RAG philosophy with inverted control

## Performance Considerations

### Current Challenges

1. **Scalability**:
   - Ultra-long dialogue management
   - Graph label maintenance (potential unlimited growth)

2. **Cost**:
   - Frequent API calls
   - Solution: Deploy self-hosted fine-tuned small models (SFT)

### Optimization Strategies

- Use domain-specific medical coding (ICD-10, SNOMED CT) for entity standardization
- Post-process to merge semantically similar labels/nodes
- Implement token-based cost estimation
- Cache frequent queries

## Future Enhancements

- [ ] Replace LLM classifiers with fine-tuned small models
- [ ] Implement precise token counting with tiktoken
- [ ] Add dialogue compression for ultra-long sessions
- [ ] Enhance graph query capabilities
- [ ] Implement update/conflict resolution for long-term memory
- [ ] Add evaluation metrics (recall, precision)
- [ ] Multi-language support

## License

MIT License

## Contributors

Project developed as part of medical AI research.

## References

- [Mem0 Platform Documentation](https://docs.mem0.ai/platform/overview)
- [Amigo.ai Memory Architecture](https://docs.amigo.ai/agent/memory/layered-architecture)

## Contact

For questions and support, please open an issue on GitHub.
