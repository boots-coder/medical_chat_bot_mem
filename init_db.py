#!/usr/bin/env python3
"""
Database Initialization Script
数据库初始化脚本

功能：
1. 创建必要的目录结构
2. 初始化 SQLite 数据库
3. 初始化 Chroma 向量数据库
4. 测试 Neo4j 连接（可选）
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.core.DatabaseManager import get_db_manager
from backend.core.config import settings


def create_directories():
    """创建必要的目录"""
    print("=" * 60)
    print("步骤1: 创建必要的目录")
    print("=" * 60)

    directories = [
        Path("data"),
        Path("data/chroma"),
        Path("logs")
    ]

    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✓ 创建目录: {directory}")
        else:
            print(f"  目录已存在: {directory}")

    print()


def initialize_databases():
    """初始化所有数据库"""
    print("=" * 60)
    print("步骤2: 初始化数据库")
    print("=" * 60)

    try:
        # 获取数据库管理器（会自动初始化所有数据库）
        db = get_db_manager()

        print("\n✓ 数据库管理器初始化成功\n")

        # 测试 SQLite
        print("【SQLite 数据库】")
        cursor = db.sqlite_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"  ✓ SQLite 已连接")
        print(f"  ✓ 表数量: {len(tables)}")
        for table in tables:
            print(f"    - {table[0]}")

        # 测试 Chroma
        print("\n【Chroma 向量数据库】")
        collection_info = db.chroma_collection.count()
        print(f"  ✓ Chroma 已连接")
        print(f"  ✓ Collection: {db.chroma_collection.name}")
        print(f"  ✓ 文档数量: {collection_info}")

        # 测试 Neo4j（可选）
        print("\n【Neo4j 图数据库】")
        if db.neo4j_driver:
            try:
                with db.neo4j_driver.session() as session:
                    result = session.run("RETURN 1 AS test")
                    test_value = result.single()[0]
                    if test_value == 1:
                        print("  ✓ Neo4j 已连接")

                        # 查询节点和关系数量
                        node_count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
                        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]

                        print(f"  ✓ 节点数量: {node_count}")
                        print(f"  ✓ 关系数量: {rel_count}")
            except Exception as e:
                print(f"  ⚠️  Neo4j 连接测试失败: {e}")
                print("  提示: Neo4j 是可选的，如果不需要知识图谱功能可以跳过")
        else:
            print("  ⚠️  Neo4j 未配置或未连接")
            print("  提示: Neo4j 是可选的，如果需要请启动 Neo4j 服务")

        print("\n✓ 所有核心数据库初始化成功！")
        return True

    except Exception as e:
        print(f"\n✗ 数据库初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def display_configuration():
    """显示当前配置"""
    print("\n" + "=" * 60)
    print("当前配置")
    print("=" * 60)

    print(f"API Provider:        {settings.api_provider}")
    print(f"API Base URL:        {settings.api_base_url}")
    print(f"Server Host:         {settings.host}")
    print(f"Server Port:         {settings.port}")
    print(f"Debug Mode:          {settings.debug}")
    print(f"\nSQLite Path:         {settings.sqlite_db_path}")
    print(f"Chroma Path:         {settings.chroma_persist_dir}")
    print(f"Neo4j URI:           {settings.neo4j_uri}")
    print(f"\nSession Timeout:     {settings.session_timeout_minutes} 分钟")
    print(f"Max Dialogue Turns:  {settings.max_dialogue_turns}")
    print()


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("医疗咨询系统 - 数据库初始化")
    print("=" * 60)
    print()

    # 显示配置
    display_configuration()

    # 创建目录
    create_directories()

    # 初始化数据库
    success = initialize_databases()

    # 最终报告
    print("\n" + "=" * 60)
    if success:
        print("✓ 初始化完成！")
        print("=" * 60)
        print("\n下一步:")
        print("1. 启动服务: python run.py")
        print("2. 访问 API 文档: http://localhost:8000/docs")
        print("3. 访问测试页面: http://localhost:8000/test")
        print()
        return 0
    else:
        print("✗ 初始化失败，请检查错误信息")
        print("=" * 60)
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
