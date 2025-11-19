#!/usr/bin/env python3
"""
Database Initialization Script

Features:
1. Create necessary directory structure
2. Initialize SQLite database
3. Initialize Chroma vector database
4. Test Neo4j connection (optional)
"""
import sys
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.core.DatabaseManager import get_db_manager
from backend.core.config import settings


def create_directories():
    """Create necessary directories"""
    print("=" * 60)
    print("Step 1: Create necessary directories")
    print("=" * 60)

    directories = [
        Path("data"),
        Path("data/chroma"),
        Path("logs")
    ]

    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✓ Directory created: {directory}")
        else:
            print(f"  Directory already exists: {directory}")

    print()


def initialize_databases():
    """Initialize all databases"""
    print("=" * 60)
    print("Step 2: Initialize databases")
    print("=" * 60)

    try:
        # Get database manager (automatically initializes all databases)
        db = get_db_manager()

        print("\n✓ Database manager initialization successful\n")

        # Test SQLite
        print("【SQLite Database】")
        cursor = db.sqlite_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"  ✓ SQLite connected")
        print(f"  ✓ Table count: {len(tables)}")
        for table in tables:
            print(f"    - {table[0]}")

        # Test Chroma
        print("\n【Chroma Vector Database】")
        collection_info = db.chroma_collection.count()
        print(f"  ✓ Chroma connected")
        print(f"  ✓ Collection: {db.chroma_collection.name}")
        print(f"  ✓ Document count: {collection_info}")

        # Test Neo4j (optional)
        print("\n【Neo4j Graph Database】")
        if db.neo4j_driver:
            try:
                with db.neo4j_driver.session() as session:
                    result = session.run("RETURN 1 AS test")
                    test_value = result.single()[0]
                    if test_value == 1:
                        print("  ✓ Neo4j connected")

                        # Query node and relationship counts
                        node_count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
                        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]

                        print(f"  ✓ Node count: {node_count}")
                        print(f"  ✓ Relationship count: {rel_count}")
            except Exception as e:
                print(f"  ⚠️  Neo4j connection test failed: {e}")
                print("  Hint: Neo4j is optional, can be skipped if knowledge graph features are not needed")
        else:
            print("  ⚠️  Neo4j not configured or not connected")
            print("  Hint: Neo4j is optional, please start Neo4j service if needed")

        print("\n✓ All core databases initialized successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def display_configuration():
    """Display current configuration"""
    print("\n" + "=" * 60)
    print("Current Configuration")
    print("=" * 60)

    print(f"API Provider:        {settings.api_provider}")
    print(f"API Base URL:        {settings.api_base_url}")
    print(f"Server Host:         {settings.host}")
    print(f"Server Port:         {settings.port}")
    print(f"Debug Mode:          {settings.debug}")
    print(f"\nSQLite Path:         {settings.sqlite_db_path}")
    print(f"Chroma Path:         {settings.chroma_persist_dir}")
    print(f"Neo4j URI:           {settings.neo4j_uri}")
    print(f"\nSession Timeout:     {settings.session_timeout_minutes} minutes")
    print(f"Max Dialogue Turns:  {settings.max_dialogue_turns}")
    print()


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("Medical Consultation System - Database Initialization")
    print("=" * 60)
    print()

    # Display configuration
    display_configuration()

    # Create directories
    create_directories()

    # Initialize databases
    success = initialize_databases()

    # Final report
    print("\n" + "=" * 60)
    if success:
        print("✓ Initialization complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Start service: python run.py")
        print("2. Access API docs: http://localhost:8000/docs")
        print("3. Access test page: http://localhost:8000/test")
        print()
        return 0
    else:
        print("✗ Initialization failed, please check error messages")
        print("=" * 60)
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
