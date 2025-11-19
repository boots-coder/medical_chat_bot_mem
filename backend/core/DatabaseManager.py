"""
数据库管理器：统一管理SQLite、Chroma、Neo4j三个数据库的连接和操作
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
    """数据库管理器：管理三个数据库的连接和基本操作"""

    def __init__(self):
        """初始化数据库连接"""
        self.sqlite_conn: Optional[sqlite3.Connection] = None
        self.chroma_client: Optional[chromadb.ClientAPI] = None
        self.chroma_collection: Optional[chromadb.Collection] = None
        self.neo4j_driver: Optional[GraphDatabase.driver] = None

        # 初始化各个数据库
        self._init_sqlite()
        self._init_chroma()
        self._init_neo4j()

    # ==================== SQLite 初始化 ====================

    def _init_sqlite(self):
        """初始化SQLite数据库"""
        try:
            # 确保数据目录存在
            db_path = Path(settings.sqlite_db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # 连接数据库
            self.sqlite_conn = sqlite3.connect(
                settings.sqlite_db_path,
                check_same_thread=False,  # 允许多线程访问
                timeout=10.0
            )
            self.sqlite_conn.row_factory = sqlite3.Row  # 使用Row对象，可以按列名访问

            # 创建表
            self.sqlite_conn.executescript(SQLITE_SCHEMA)
            self.sqlite_conn.commit()

            print(f"✓ SQLite 初始化成功: {settings.sqlite_db_path}")

        except Exception as e:
            print(f"✗ SQLite 初始化失败: {e}")
            raise

    # ==================== Chroma 初始化 ====================

    def _init_chroma(self):
        """初始化Chroma向量数据库"""
        try:
            # 确保持久化目录存在
            persist_dir = Path(settings.chroma_persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)

            # 创建Chroma客户端（持久化模式）
            self.chroma_client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # 获取或创建collection
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="medical_memory",
                metadata={"description": "医疗对话长期记忆存储"}
            )

            print(f"✓ Chroma 初始化成功: {settings.chroma_persist_dir}")
            print(f"  Collection: medical_memory, 文档数: {self.chroma_collection.count()}")

        except Exception as e:
            print(f"✗ Chroma 初始化失败: {e}")
            raise

    # ==================== Neo4j 初始化 ====================

    def _init_neo4j(self):
        """初始化Neo4j图数据库"""
        try:
            # 创建Neo4j驱动
            self.neo4j_driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password)
            )

            # 验证连接
            self.neo4j_driver.verify_connectivity()

            # 创建约束和索引
            with self.neo4j_driver.session() as session:
                # 创建唯一性约束
                for constraint in NEO4J_SCHEMA["constraints"]:
                    try:
                        session.run(constraint)
                    except Exception as e:
                        # 约束可能已存在，忽略错误
                        pass

                # 创建索引
                for index in NEO4J_SCHEMA["indexes"]:
                    try:
                        session.run(index)
                    except Exception as e:
                        # 索引可能已存在，忽略错误
                        pass

            print(f"✓ Neo4j 初始化成功: {settings.neo4j_uri}")

        except Exception as e:
            print(f"✗ Neo4j 初始化失败: {e}")
            print(f"  提示: 请确保Neo4j服务正在运行，用户名密码正确")
            # Neo4j不是必需的，可以继续运行
            self.neo4j_driver = None

    # ==================== 会话管理（SQLite）====================

    def create_session(
        self,
        session_id: str,
        patient_id: str,
        url_token: str,
        token_expires_at: datetime,
        patient_info: Dict[str, Any]
    ) -> bool:
        """创建新会话"""
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
            print(f"创建会话失败（可能已存在）: {e}")
            return False

    def get_session_by_token(self, url_token: str) -> Optional[Dict]:
        """通过token获取会话信息"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE url_token = ?", (url_token,))
        row = cursor.fetchone()

        if row:
            return dict(row)
        return None

    def update_session_activity(self, session_id: str):
        """更新会话活动时间"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute("""
            UPDATE sessions
            SET last_activity_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
        """, (session_id,))
        self.sqlite_conn.commit()

    def end_session(self, session_id: str):
        """结束会话"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute("""
            UPDATE sessions
            SET status = 'ended',
                ended_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
        """, (session_id,))
        self.sqlite_conn.commit()

    def get_expired_sessions(self) -> List[Dict]:
        """获取所有超时的会话"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute("""
            SELECT * FROM sessions
            WHERE status = 'active'
              AND datetime(token_expires_at) < datetime('now')
        """)
        return [dict(row) for row in cursor.fetchall()]

    # ==================== 向量存储（Chroma）====================

    def store_memory_unit(
        self,
        unit_id: str,
        embedding: List[float],
        document: str,
        metadata: Dict[str, Any]
    ):
        """存储记忆单元到向量数据库"""
        try:
            self.chroma_collection.add(
                ids=[unit_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata]
            )
            print(f"✓ 向量存储成功: {unit_id}")

        except Exception as e:
            print(f"✗ 向量存储失败: {e}")
            raise

    def query_memory_by_vector(
        self,
        query_embedding: List[float],
        patient_id: str,
        n_results: int = 5,
        additional_filters: Optional[Dict] = None
    ) -> Dict:
        """通过向量相似度查询记忆"""
        try:
            # 构建过滤条件
            where_filter = {"patient_id": patient_id}

            if additional_filters:
                # 如果有额外过滤条件，使用$and
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
            print(f"✗ 向量查询失败: {e}")
            return {"ids": [[]], "metadatas": [[]], "documents": [[]]}

    # ==================== 图存储（Neo4j）====================

    def store_knowledge_graph(
        self,
        patient_id: str,
        session_id: str,
        knowledge_graph: Dict[str, Any],
        timestamp: str
    ):
        """存储知识图谱到Neo4j"""
        if not self.neo4j_driver:
            print("⚠️  Neo4j未连接，跳过图存储")
            return

        try:
            with self.neo4j_driver.session() as session:
                # 1. 确保患者节点存在
                session.run(
                    "MERGE (p:Patient {patient_id: $patient_id})",
                    patient_id=patient_id
                )

                # 2. 创建实体节点
                for entity in knowledge_graph.get('entities', []):
                    label = entity['type']
                    entity_id = entity['id']
                    name = entity.get('label', entity.get('name', ''))

                    session.run(f"""
                        MERGE (e:{label} {{id: $id}})
                        SET e.name = $name
                    """, id=entity_id, name=name)

                # 3. 创建关系
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

            print(f"✓ 知识图谱存储成功: session={session_id}")

        except Exception as e:
            print(f"✗ 知识图谱存储失败: {e}")
            # 图存储失败不影响主流程

    def query_graph(
        self,
        query_type: str,
        patient_id: str,
        additional_params: Optional[Dict] = None
    ) -> List[Dict]:
        """执行图查询"""
        if not self.neo4j_driver:
            print("⚠️  Neo4j未连接，返回空结果")
            return []

        try:
            # 获取查询模板
            cypher = GRAPH_QUERY_TEMPLATES.get(query_type)
            if not cypher:
                print(f"✗ 未找到查询模板: {query_type}")
                return []

            # 执行查询
            params = {"patient_id": patient_id}
            if additional_params:
                params.update(additional_params)

            with self.neo4j_driver.session() as session:
                result = session.run(cypher, **params)
                records = [dict(record) for record in result]

            return records

        except Exception as e:
            print(f"✗ 图查询失败: {e}")
            return []

    # ==================== 清理和关闭 ====================

    def close(self):
        """关闭所有数据库连接"""
        if self.sqlite_conn:
            self.sqlite_conn.close()
            print("✓ SQLite 连接已关闭")

        if self.neo4j_driver:
            self.neo4j_driver.close()
            print("✓ Neo4j 连接已关闭")

        # Chroma客户端不需要显式关闭

    def __enter__(self):
        """上下文管理器支持"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()


# 全局数据库管理器实例
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """获取全局数据库管理器实例（单例模式）"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


if __name__ == "__main__":
    print("=== 数据库管理器测试 ===\n")

    # 测试初始化
    with DatabaseManager() as db:
        print("\n数据库初始化完成")

        # 测试SQLite
        print("\n--- 测试SQLite ---")
        print(f"会话表是否存在: ", end="")
        cursor = db.sqlite_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
        print("✓" if cursor.fetchone() else "✗")

        # 测试Chroma
        print("\n--- 测试Chroma ---")
        print(f"Collection文档数: {db.chroma_collection.count()}")

        # 测试Neo4j
        print("\n--- 测试Neo4j ---")
        if db.neo4j_driver:
            with db.neo4j_driver.session() as session:
                result = session.run("RETURN 1 AS test")
                print(f"连接测试: {'✓' if result.single()['test'] == 1 else '✗'}")
        else:
            print("Neo4j未连接")

    print("\n数据库连接已关闭")
