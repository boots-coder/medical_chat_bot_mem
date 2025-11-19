#!/usr/bin/env python3
"""
批量更新 backend 目录下所有Python文件的import语句
将相对导入改为绝对导入（基于backend包）
"""
import os
import re
from pathlib import Path

# 定义import映射规则
IMPORT_MAPPINGS = {
    # Core
    'from config import': 'from backend.core.config import',
    'from DatabaseManager import': 'from backend.core.DatabaseManager import',
    'from database_schemas import': 'from backend.core.database_schemas import',

    # Services
    'from SessionManager import': 'from backend.services.SessionManager import',
    'from TokenManager import': 'from backend.services.TokenManager import',
    'from MemoryStorage import': 'from backend.services.MemoryStorage import',
    'from MemoryRetrieval import': 'from backend.services.MemoryRetrieval import',
    'from MedicalResponseGenerator import': 'from backend.services.MedicalResponseGenerator import',
    'from DialogueAnalyzer import': 'from backend.services.DialogueAnalyzer import',

    # Models
    'from ShortTermMemoryManager import': 'from backend.models.ShortTermMemoryManager import',

    # ML
    'from APIManager import': 'from backend.ml.APIManager import',
    'from RAGIntentClassifier import': 'from backend.ml.RAGIntentClassifier import',
    'from LightweightMedicalClassifier import': 'from backend.ml.LightweightMedicalClassifier import',
    'from context_aware_clusterer import': 'from backend.ml.context_aware_clusterer import',
}

def update_file_imports(file_path: Path):
    """更新单个文件的imports"""
    print(f"处理: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    changed = False

    # 应用所有映射规则
    for old_import, new_import in IMPORT_MAPPINGS.items():
        if old_import in content:
            content = content.replace(old_import, new_import)
            changed = True
            print(f"  ✓ 替换: {old_import} → {new_import}")

    if changed:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ 文件已更新")
    else:
        print(f"  - 无需更新")

    return changed

def main():
    """主函数"""
    backend_dir = Path(__file__).parent / 'backend'

    if not backend_dir.exists():
        print(f"错误：backend目录不存在: {backend_dir}")
        return

    print("开始批量更新import语句...\n")

    # 查找所有Python文件
    py_files = list(backend_dir.rglob('*.py'))
    py_files = [f for f in py_files if f.name != '__init__.py']

    print(f"找到 {len(py_files)} 个Python文件\n")

    updated_count = 0
    for py_file in sorted(py_files):
        if update_file_imports(py_file):
            updated_count += 1
        print()

    print(f"\n完成！共更新了 {updated_count} 个文件")

if __name__ == '__main__':
    main()
