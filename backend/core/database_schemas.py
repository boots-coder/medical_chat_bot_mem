"""
数据库Schema定义
包含SQLite、Chroma、Neo4j三个数据库的结构定义
"""

# ==================== SQLite Schema ====================

SQLITE_SCHEMA = """
-- 会话管理表（仅存储临时会话状态）
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    url_token TEXT UNIQUE NOT NULL,
    token_expires_at TIMESTAMP NOT NULL,

    -- 会话状态
    status TEXT DEFAULT 'active' CHECK(status IN ('active', 'ended', 'expired')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,

    -- 外部医疗系统提供的临时信息
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
    "description": "向量数据库metadata结构定义",
    "fields": {
        # 核心标识字段
        "patient_id": {
            "type": "str",
            "required": True,
            "description": "患者ID，用于过滤查询"
        },
        "unit_type": {
            "type": "str",
            "required": True,
            "enum": ["session", "cluster"],
            "description": "记忆单元类型"
        },
        "session_id": {
            "type": "str",
            "required": True,
            "description": "原始会话ID"
        },
        "cluster_id": {
            "type": "int",
            "required": False,
            "description": "聚类簇ID（仅cluster类型有值）"
        },

        # 时间字段
        "created_at": {
            "type": "str",  # ISO 8601格式
            "required": True,
            "description": "创建时间"
        },
        "end_time": {
            "type": "str",
            "required": True,
            "description": "对话结束时间"
        },

        # 完整分析结果（JSON字符串）
        "analysis_json": {
            "type": "str",  # JSON.dumps()的结果
            "required": True,
            "description": "DialogueAnalyzer的完整输出",
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
        "analysis_json": '{"session_topic": "头痛咨询", "narrative_summary": "...", ...}'
    }
}

# ==================== Neo4j Schema ====================

NEO4J_SCHEMA = {
    "description": "图数据库schema定义",

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
                "id": {"type": "str", "unique": True, "required": True},  # 格式: S_symptom_name
                "name": {"type": "str", "required": True}
            },
            "indexes": ["id", "name"]
        },

        "Disease": {
            "label": "Disease",
            "properties": {
                "id": {"type": "str", "unique": True, "required": True},  # 格式: D_disease_name
                "name": {"type": "str", "required": True}
            },
            "indexes": ["id", "name"]
        },

        "Diagnosis": {
            "label": "Diagnosis",
            "properties": {
                "id": {"type": "str", "unique": True, "required": True},  # 格式: DG_diagnosis_name
                "name": {"type": "str", "required": True}
            },
            "indexes": ["id", "name"]
        },

        "Drug": {
            "label": "Drug",
            "properties": {
                "id": {"type": "str", "unique": True, "required": True},  # 格式: DR_drug_name
                "name": {"type": "str", "required": True}
            },
            "indexes": ["id", "name"]
        },

        "Examination": {
            "label": "Examination",
            "properties": {
                "id": {"type": "str", "unique": True, "required": True},  # 格式: E_exam_name
                "name": {"type": "str", "required": True}
            },
            "indexes": ["id", "name"]
        },

        "Treatment": {
            "label": "Treatment",
            "properties": {
                "id": {"type": "str", "unique": True, "required": True},  # 格式: T_treatment_name
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

# ==================== 图查询模板 ====================

GRAPH_QUERY_TEMPLATES = {
    "drug_interaction": """
        // 查询患者当前用药的相互作用
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
        // 查询患者症状可能关联的疾病
        MATCH (p:Patient {patient_id: $patient_id})-[:HAS_SYMPTOM]->(s:Symptom)
        MATCH (s)<-[:MAY_CAUSE]-(d:Disease)
        RETURN s.name AS symptom,
               d.name AS disease,
               COUNT(*) AS occurrence_count
        ORDER BY occurrence_count DESC
        LIMIT 10
    """,

    "diagnosis_chain": """
        // 查询患者的诊断链（症状→诊断→治疗）
        MATCH path = (p:Patient {patient_id: $patient_id})-[:HAS_SYMPTOM]->(s:Symptom)
                     -[:MAY_CAUSE]->(d:Disease)
                     <-[:IS_SUGGESTED_FOR]-(dg:Diagnosis)
        RETURN path
        ORDER BY dg.suggested_at DESC
        LIMIT 5
    """,

    "treatment_history": """
        // 查询患者的治疗历史
        MATCH (p:Patient {patient_id: $patient_id})-[r:PRESCRIBED]->(dr:Drug)
        RETURN dr.name AS drug,
               r.prescribed_at AS prescribed_at,
               r.dosage AS dosage
        ORDER BY r.prescribed_at DESC
        LIMIT 10
    """,

    "patient_medical_graph": """
        // 获取患者的完整医疗图谱（1度关系）
        MATCH (p:Patient {patient_id: $patient_id})-[r]->(n)
        RETURN p, r, n
        LIMIT 100
    """
}


if __name__ == "__main__":
    print("=== 数据库Schema定义 ===\n")

    print("1. SQLite Schema:")
    print(SQLITE_SCHEMA)

    print("\n2. Chroma Metadata Schema:")
    print(f"  字段数: {len(CHROMA_METADATA_SCHEMA['fields'])}")
    print(f"  示例: {CHROMA_METADATA_SCHEMA['example']}")

    print("\n3. Neo4j Schema:")
    print(f"  节点类型: {len(NEO4J_SCHEMA['nodes'])}")
    print(f"  关系类型: {len(NEO4J_SCHEMA['relationships'])}")
    print(f"  约束数量: {len(NEO4J_SCHEMA['constraints'])}")

    print("\n4. 图查询模板:")
    for query_type in GRAPH_QUERY_TEMPLATES.keys():
        print(f"  - {query_type}")
