"""
Database Schema Definitions
Contains structure definitions for SQLite, Chroma, and Neo4j databases
"""

# ==================== SQLite Schema ====================

SQLITE_SCHEMA = """
-- Session management table (stores only temporary session state)
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    url_token TEXT UNIQUE NOT NULL,
    token_expires_at TIMESTAMP NOT NULL,

    -- Session status
    status TEXT DEFAULT 'active' CHECK(status IN ('active', 'ended', 'expired')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,

    -- Temporary information provided by external medical system
    patient_name TEXT,
    patient_age INTEGER,
    gender TEXT,
    doctor_name TEXT,
    department TEXT,
    appointment_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(url_token);
CREATE INDEX IF NOT EXISTS idx_sessions_patient ON sessions(patient_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(token_expires_at);
"""

# ==================== Chroma Metadata Schema ====================

CHROMA_METADATA_SCHEMA = {
    "description": "Vector database metadata structure definition",
    "fields": {
        # Core identifier fields
        "patient_id": {
            "type": "str",
            "required": True,
            "description": "Patient ID, used for filtering queries"
        },
        "unit_type": {
            "type": "str",
            "required": True,
            "enum": ["session", "cluster"],
            "description": "Memory unit type"
        },
        "session_id": {
            "type": "str",
            "required": True,
            "description": "Original session ID"
        },
        "cluster_id": {
            "type": "int",
            "required": False,
            "description": "Cluster ID (only for cluster type)"
        },

        # Time fields
        "created_at": {
            "type": "str",  # ISO 8601 format
            "required": True,
            "description": "Creation time"
        },
        "end_time": {
            "type": "str",
            "required": True,
            "description": "Dialogue end time"
        },

        # Complete analysis results (JSON string)
        "analysis_json": {
            "type": "str",  # Result of JSON.dumps()
            "required": True,
            "description": "Complete output from DialogueAnalyzer",
            "contains": {
                "session_topic": "str",
                "narrative_summary": "str",
                "main_complaint_vectorized": "str",
                "knowledge_graph": {
                    "entities": "list",
                    "relationships": "list"
                },
                "dialogue_rounds": "int"
            }
        }
    },

    "example": {
        "patient_id": "P12345",
        "unit_type": "session",
        "session_id": "S001",
        "cluster_id": None,
        "created_at": "2025-11-19T10:00:00Z",
        "end_time": "2025-11-19T10:15:00Z",
        "analysis_json": '{"session_topic": "Headache consultation", "narrative_summary": "...", ...}'
    }
}

# ==================== Neo4j Schema ====================

NEO4J_SCHEMA = {
    "description": "Graph database schema definition",

    "nodes": {
        "Patient": {
            "label": "Patient",
            "properties": {
                "patient_id": {"type": "str", "unique": True, "required": True}
            },
            "indexes": ["patient_id"]
        },

        "Symptom": {
            "label": "Symptom",
            "properties": {
                "id": {"type": "str", "unique": True, "required": True},  # Format: S_symptom_name
                "name": {"type": "str", "required": True}
            },
            "indexes": ["id", "name"]
        },

        "Disease": {
            "label": "Disease",
            "properties": {
                "id": {"type": "str", "unique": True, "required": True},  # Format: D_disease_name
                "name": {"type": "str", "required": True}
            },
            "indexes": ["id", "name"]
        },

        "Diagnosis": {
            "label": "Diagnosis",
            "properties": {
                "id": {"type": "str", "unique": True, "required": True},  # Format: DG_diagnosis_name
                "name": {"type": "str", "required": True}
            },
            "indexes": ["id", "name"]
        },

        "Drug": {
            "label": "Drug",
            "properties": {
                "id": {"type": "str", "unique": True, "required": True},  # Format: DR_drug_name
                "name": {"type": "str", "required": True}
            },
            "indexes": ["id", "name"]
        },

        "Examination": {
            "label": "Examination",
            "properties": {
                "id": {"type": "str", "unique": True, "required": True},  # Format: E_exam_name
                "name": {"type": "str", "required": True}
            },
            "indexes": ["id", "name"]
        },

        "Treatment": {
            "label": "Treatment",
            "properties": {
                "id": {"type": "str", "unique": True, "required": True},  # Format: T_treatment_name
                "name": {"type": "str", "required": True}
            },
            "indexes": ["id", "name"]
        }
    },

    "relationships": {
        "HAS_SYMPTOM": {
            "from": "Patient",
            "to": "Symptom",
            "properties": {
                "session_id": {"type": "str", "required": True},
                "observed_at": {"type": "str", "required": True}  # ISO 8601
            }
        },

        "HAS_HISTORY": {
            "from": "Patient",
            "to": "Disease",
            "properties": {
                "session_id": {"type": "str", "required": True},
                "diagnosed_at": {"type": "str", "required": False}
            }
        },

        "MAY_CAUSE": {
            "from": "Disease",
            "to": "Symptom",
            "properties": {
                "session_id": {"type": "str", "required": True},
                "probability": {"type": "float", "required": False}
            }
        },

        "IS_SUGGESTED_FOR": {
            "from": "Diagnosis",
            "to": "Patient",
            "properties": {
                "session_id": {"type": "str", "required": True},
                "suggested_at": {"type": "str", "required": True}
            }
        },

        "RECOMMENDED_FOR": {
            "from": "Drug",
            "to": "Diagnosis",
            "properties": {
                "session_id": {"type": "str", "required": True},
                "prescribed_at": {"type": "str", "required": True}
            }
        },

        "PRESCRIBED": {
            "from": "Patient",
            "to": "Drug",
            "properties": {
                "session_id": {"type": "str", "required": True},
                "prescribed_at": {"type": "str", "required": True},
                "dosage": {"type": "str", "required": False}
            }
        },

        "REQUIRES": {
            "from": "Diagnosis",
            "to": "Examination",
            "properties": {
                "session_id": {"type": "str", "required": True}
            }
        },

        "INDICATES": {
            "from": "Examination",
            "to": "Disease",
            "properties": {
                "session_id": {"type": "str", "required": True},
                "result": {"type": "str", "required": False}
            }
        },

        "INTERACTS_WITH": {
            "from": "Drug",
            "to": "Drug",
            "properties": {
                "severity": {"type": "str", "required": False},  # "mild", "moderate", "severe"
                "description": {"type": "str", "required": False}
            }
        }
    },

    "constraints": [
        "CREATE CONSTRAINT patient_id_unique IF NOT EXISTS FOR (p:Patient) REQUIRE p.patient_id IS UNIQUE",
        "CREATE CONSTRAINT symptom_id_unique IF NOT EXISTS FOR (s:Symptom) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT disease_id_unique IF NOT EXISTS FOR (d:Disease) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT diagnosis_id_unique IF NOT EXISTS FOR (dg:Diagnosis) REQUIRE dg.id IS UNIQUE",
        "CREATE CONSTRAINT drug_id_unique IF NOT EXISTS FOR (dr:Drug) REQUIRE dr.id IS UNIQUE",
        "CREATE CONSTRAINT exam_id_unique IF NOT EXISTS FOR (e:Examination) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT treatment_id_unique IF NOT EXISTS FOR (t:Treatment) REQUIRE t.id IS UNIQUE"
    ],

    "indexes": [
        "CREATE INDEX patient_id_idx IF NOT EXISTS FOR (p:Patient) ON (p.patient_id)",
        "CREATE INDEX symptom_name_idx IF NOT EXISTS FOR (s:Symptom) ON (s.name)",
        "CREATE INDEX disease_name_idx IF NOT EXISTS FOR (d:Disease) ON (d.name)",
        "CREATE INDEX drug_name_idx IF NOT EXISTS FOR (dr:Drug) ON (dr.name)"
    ]
}

# ==================== Graph Query Templates ====================

GRAPH_QUERY_TEMPLATES = {
    "drug_interaction": """
        // Query patient's current medication interactions
        MATCH (p:Patient {patient_id: $patient_id})-[:PRESCRIBED]->(dr1:Drug)
        MATCH (dr1)-[i:INTERACTS_WITH]->(dr2:Drug)
        RETURN dr1.name AS drug1,
               dr2.name AS drug2,
               i.severity AS severity,
               i.description AS description
        ORDER BY
            CASE i.severity
                WHEN 'severe' THEN 1
                WHEN 'moderate' THEN 2
                WHEN 'mild' THEN 3
                ELSE 4
            END
    """,

    "symptom_disease": """
        // Query diseases potentially related to patient's symptoms
        MATCH (p:Patient {patient_id: $patient_id})-[:HAS_SYMPTOM]->(s:Symptom)
        MATCH (s)<-[:MAY_CAUSE]-(d:Disease)
        RETURN s.name AS symptom,
               d.name AS disease,
               COUNT(*) AS occurrence_count
        ORDER BY occurrence_count DESC
        LIMIT 10
    """,

    "diagnosis_chain": """
        // Query patient's diagnosis chain (symptom→diagnosis→treatment)
        MATCH path = (p:Patient {patient_id: $patient_id})-[:HAS_SYMPTOM]->(s:Symptom)
                     -[:MAY_CAUSE]->(d:Disease)
                     <-[:IS_SUGGESTED_FOR]-(dg:Diagnosis)
        RETURN path
        ORDER BY dg.suggested_at DESC
        LIMIT 5
    """,

    "treatment_history": """
        // Query patient's treatment history
        MATCH (p:Patient {patient_id: $patient_id})-[r:PRESCRIBED]->(dr:Drug)
        RETURN dr.name AS drug,
               r.prescribed_at AS prescribed_at,
               r.dosage AS dosage
        ORDER BY r.prescribed_at DESC
        LIMIT 10
    """,

    "patient_medical_graph": """
        // Get patient's complete medical graph (1-degree relationships)
        MATCH (p:Patient {patient_id: $patient_id})-[r]->(n)
        RETURN p, r, n
        LIMIT 100
    """
}


if __name__ == "__main__":
    print("=== Database Schema Definitions ===\n")

    print("1. SQLite Schema:")
    print(SQLITE_SCHEMA)

    print("\n2. Chroma Metadata Schema:")
    print(f"  Field count: {len(CHROMA_METADATA_SCHEMA['fields'])}")
    print(f"  Example: {CHROMA_METADATA_SCHEMA['example']}")

    print("\n3. Neo4j Schema:")
    print(f"  Node types: {len(NEO4J_SCHEMA['nodes'])}")
    print(f"  Relationship types: {len(NEO4J_SCHEMA['relationships'])}")
    print(f"  Constraint count: {len(NEO4J_SCHEMA['constraints'])}")

    print("\n4. Graph Query Templates:")
    for query_type in GRAPH_QUERY_TEMPLATES.keys():
        print(f"  - {query_type}")
