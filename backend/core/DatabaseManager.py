"""
Database Manager: Unified management of SQLite, Chroma, and Neo4j database connections and operations
"""
import sqlite3
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from neo4j import GraphDatabase

from backend.core.config import settings
from backend.core.database_schemas import SQLITE_SCHEMA, NEO4J_SCHEMA, GRAPH_QUERY_TEMPLATES


class DatabaseManager:
    """Database Manager: Manages connections and basic operations for three databases"""

    def __init__(self):
        """Initialize database connections"""
        self.sqlite_conn: Optional[sqlite3.Connection] = None
        self.chroma_client: Optional[chromadb.ClientAPI] = None
        self.chroma_collection: Optional[chromadb.Collection] = None
        self.neo4j_driver: Optional[GraphDatabase.driver] = None

        # Initialize all databases
        self._init_sqlite()
        self._init_chroma()
        self._init_neo4j()

    # ==================== SQLite Initialization ====================

    def _init_sqlite(self):
        """Initialize SQLite database"""
        try:
            # Ensure data directory exists
            db_path = Path(settings.sqlite_db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # Connect to database
            self.sqlite_conn = sqlite3.connect(
                settings.sqlite_db_path,
                check_same_thread=False,  # Allow multi-threaded access
                timeout=10.0
            )
            self.sqlite_conn.row_factory = sqlite3.Row  # Use Row objects for column name access

            # Create tables
            self.sqlite_conn.executescript(SQLITE_SCHEMA)
            self.sqlite_conn.commit()

            print(f"✓ SQLite initialization successful: {settings.sqlite_db_path}")

        except Exception as e:
            print(f"✗ SQLite initialization failed: {e}")
            raise

    # ==================== Chroma Initialization ====================

    def _init_chroma(self):
        """Initialize Chroma vector database"""
        try:
            # Ensure persistence directory exists
            persist_dir = Path(settings.chroma_persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)

            # Create Chroma client (persistent mode)
            self.chroma_client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Get or create collection
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="medical_memory",
                metadata={"description": "Medical dialogue long-term memory storage"}
            )

            print(f"✓ Chroma initialization successful: {settings.chroma_persist_dir}")
            print(f"  Collection: medical_memory, Document count: {self.chroma_collection.count()}")

        except Exception as e:
            print(f"✗ Chroma initialization failed: {e}")
            raise

    # ==================== Neo4j Initialization ====================

    def _init_neo4j(self):
        """Initialize Neo4j graph database"""
        try:
            # Create Neo4j driver
            self.neo4j_driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password)
            )

            # Verify connectivity
            self.neo4j_driver.verify_connectivity()

            # Create constraints and indexes
            with self.neo4j_driver.session() as session:
                # Create uniqueness constraints
                for constraint in NEO4J_SCHEMA["constraints"]:
                    try:
                        session.run(constraint)
                    except Exception as e:
                        # Constraint may already exist, ignore error
                        pass

                # Create indexes
                for index in NEO4J_SCHEMA["indexes"]:
                    try:
                        session.run(index)
                    except Exception as e:
                        # Index may already exist, ignore error
                        pass

            print(f"✓ Neo4j initialization successful: {settings.neo4j_uri}")

        except Exception as e:
            print(f"✗ Neo4j initialization failed: {e}")
            print(f"  Hint: Please ensure Neo4j service is running and credentials are correct")
            # Neo4j is not required, can continue running
            self.neo4j_driver = None

    # ==================== Session Management (SQLite) ====================

    def create_session(
        self,
        session_id: str,
        patient_id: str,
        url_token: str,
        token_expires_at: datetime,
        patient_info: Dict[str, Any]
    ) -> bool:
        """Create a new session"""
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (
                    session_id, patient_id, url_token, token_expires_at,
                    patient_name, patient_age, gender, doctor_name, department, appointment_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                patient_id,
                url_token,
                token_expires_at.isoformat(),
                patient_info.get('patient_name'),
                patient_info.get('patient_age'),
                patient_info.get('gender'),
                patient_info.get('doctor_name'),
                patient_info.get('department'),
                patient_info.get('appointment_id')
            ))
            self.sqlite_conn.commit()
            return True

        except sqlite3.IntegrityError as e:
            print(f"Failed to create session (may already exist): {e}")
            return False

    def get_session_by_token(self, url_token: str) -> Optional[Dict]:
        """Get session information by token"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE url_token = ?", (url_token,))
        row = cursor.fetchone()

        if row:
            return dict(row)
        return None

    def update_session_activity(self, session_id: str):
        """Update session activity time"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute("""
            UPDATE sessions
            SET last_activity_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
        """, (session_id,))
        self.sqlite_conn.commit()

    def end_session(self, session_id: str):
        """End a session"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute("""
            UPDATE sessions
            SET status = 'ended',
                ended_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
        """, (session_id,))
        self.sqlite_conn.commit()

    def get_expired_sessions(self) -> List[Dict]:
        """Get all expired sessions"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute("""
            SELECT * FROM sessions
            WHERE status = 'active'
              AND datetime(token_expires_at) < datetime('now')
        """)
        return [dict(row) for row in cursor.fetchall()]

    # ==================== Vector Storage (Chroma) ====================

    def store_memory_unit(
        self,
        unit_id: str,
        embedding: List[float],
        document: str,
        metadata: Dict[str, Any]
    ):
        """Store memory unit to vector database"""
        try:
            self.chroma_collection.add(
                ids=[unit_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata]
            )
            print(f"✓ Vector storage successful: {unit_id}")

        except Exception as e:
            print(f"✗ Vector storage failed: {e}")
            raise

    def query_memory_by_vector(
        self,
        query_embedding: List[float],
        patient_id: str,
        n_results: int = 5,
        additional_filters: Optional[Dict] = None
    ) -> Dict:
        """Query memory by vector similarity"""
        try:
            # Build filter conditions
            where_filter = {"patient_id": patient_id}

            if additional_filters:
                # If additional filters exist, use $and
                where_filter = {
                    "$and": [
                        {"patient_id": patient_id},
                        additional_filters
                    ]
                }

            results = self.chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter
            )

            return results

        except Exception as e:
            print(f"✗ Vector query failed: {e}")
            return {"ids": [[]], "metadatas": [[]], "documents": [[]]}

    # ==================== Graph Storage (Neo4j) ====================

    def store_knowledge_graph(
        self,
        patient_id: str,
        session_id: str,
        knowledge_graph: Dict[str, Any],
        timestamp: str
    ):
        """Store knowledge graph to Neo4j"""
        if not self.neo4j_driver:
            print("⚠️  Neo4j not connected, skipping graph storage")
            return

        try:
            with self.neo4j_driver.session() as session:
                # 1. Ensure patient node exists
                session.run(
                    "MERGE (p:Patient {patient_id: $patient_id})",
                    patient_id=patient_id
                )

                # 2. Create entity nodes
                for entity in knowledge_graph.get('entities', []):
                    label = entity['type']
                    entity_id = entity['id']
                    name = entity.get('label', entity.get('name', ''))

                    session.run(f"""
                        MERGE (e:{label} {{id: $id}})
                        SET e.name = $name
                    """, id=entity_id, name=name)

                # 3. Create relationships
                for rel in knowledge_graph.get('relationships', []):
                    subject_id = rel['subject']
                    object_id = rel['object']
                    predicate = rel['predicate']

                    session.run(f"""
                        MATCH (source {{id: $source_id}})
                        MATCH (target {{id: $target_id}})
                        MERGE (source)-[r:{predicate}]->(target)
                        SET r.session_id = $session_id,
                            r.created_at = $timestamp
                    """,
                    source_id=subject_id,
                    target_id=object_id,
                    session_id=session_id,
                    timestamp=timestamp
                    )

            print(f"✓ Knowledge graph storage successful: session={session_id}")

        except Exception as e:
            print(f"✗ Knowledge graph storage failed: {e}")
            # Graph storage failure doesn't affect main process

    def query_graph(
        self,
        query_type: str,
        patient_id: str,
        additional_params: Optional[Dict] = None
    ) -> List[Dict]:
        """Execute graph query"""
        if not self.neo4j_driver:
            print("⚠️  Neo4j not connected, returning empty results")
            return []

        try:
            # Get query template
            cypher = GRAPH_QUERY_TEMPLATES.get(query_type)
            if not cypher:
                print(f"✗ Query template not found: {query_type}")
                return []

            # Execute query
            params = {"patient_id": patient_id}
            if additional_params:
                params.update(additional_params)

            with self.neo4j_driver.session() as session:
                result = session.run(cypher, **params)
                records = [dict(record) for record in result]

            return records

        except Exception as e:
            print(f"✗ Graph query failed: {e}")
            return []

    # ==================== Cleanup and Shutdown ====================

    def close(self):
        """Close all database connections"""
        if self.sqlite_conn:
            self.sqlite_conn.close()
            print("✓ SQLite connection closed")

        if self.neo4j_driver:
            self.neo4j_driver.close()
            print("✓ Neo4j connection closed")

        # Chroma client doesn't need explicit closing

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get global database manager instance (singleton pattern)"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


if __name__ == "__main__":
    print("=== Database Manager Test ===\n")

    # Test initialization
    with DatabaseManager() as db:
        print("\nDatabase initialization complete")

        # Test SQLite
        print("\n--- Test SQLite ---")
        print(f"Sessions table exists: ", end="")
        cursor = db.sqlite_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
        print("✓" if cursor.fetchone() else "✗")

        # Test Chroma
        print("\n--- Test Chroma ---")
        print(f"Collection document count: {db.chroma_collection.count()}")

        # Test Neo4j
        print("\n--- Test Neo4j ---")
        if db.neo4j_driver:
            with db.neo4j_driver.session() as session:
                result = session.run("RETURN 1 AS test")
                print(f"Connection test: {'✓' if result.single()['test'] == 1 else '✗'}")
        else:
            print("Neo4j not connected")

    print("\nDatabase connections closed")
