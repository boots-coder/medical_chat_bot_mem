"""
RAG意图分类器训练脚本（支持中英文混合）
使用多语言预训练模型进行微调
"""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import torch
import argparse
import random
import numpy as np
from pathlib import Path
from datetime import datetime

from training.local_classifier import LocalRAGIntentClassifier


def set_seed(seed=42):
    """设置随机种子保证可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(data_path):
    """加载训练数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ 加载数据: {data_path}")
    print(f"  样本数量: {len(data)}")

    # 统计
    need_rag = sum(1 for s in data if s['need_rag'])
    print(f"  need_rag=True: {need_rag} ({need_rag/len(data)*100:.1f}%)")
    print(f"  need_rag=False: {len(data)-need_rag} ({(len(data)-need_rag)/len(data)*100:.1f}%)")

    return data


def main(args):
    print("="*60)
    print("RAG意图分类器训练")
    print("="*60)
    print(f"模型: {args.model}")
    print(f"训练数据: {args.train_data}")
    print(f"测试数据: {args.test_data}")
    print(f"输出目录: {args.output_dir}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print("="*60)

    # 设置随机种子
    set_seed(args.seed)

    # 加载数据
    train_samples = load_dataset(args.train_data)
    test_samples = load_dataset(args.test_data) if args.test_data else None

    # 创建分类器
    print(f"\n初始化模型...")
    classifier = LocalRAGIntentClassifier(
        model_name=args.model,
        device=args.device
    )

    # 训练
    print(f"\n开始训练...")
    history = classifier.train(
        train_samples=train_samples,
        eval_samples=test_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        output_dir=args.output_dir
    )

    # 保存训练历史
    history_file = Path(args.output_dir) / "training_history.json"
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 训练历史已保存: {history_file}")

    # 最终评估
    if test_samples:
        print("\n" + "="*60)
        print("最终评估")
        print("="*60)

        from torch.utils.data import DataLoader
        from training.local_classifier import RAGIntentDataset

        test_dataset = RAGIntentDataset(test_samples, classifier.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        metrics = classifier.evaluate(test_loader)

        print(f"测试集性能:")
        print(f"  准确率 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"  精确率 (Precision): {metrics['precision']:.4f}")
        print(f"  召回率 (Recall): {metrics['recall']:.4f}")
        print(f"  F1分数 (F1-score): {metrics['f1']:.4f}")

        # 保存评估结果
        eval_file = Path(args.output_dir) / "eval_results.json"
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 评估结果已保存: {eval_file}")

    print("\n" + "="*60)
    print("训练完成！")
    print(f"模型保存在: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练RAG意图分类器")

    # 数据参数
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="训练数据路径 (JSON格式)"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="测试数据路径 (JSON格式，可选)"
    )

    # 模型参数
    parser.add_argument(
        "--model",
        type=str,
        default="xlm-roberta-base",
        choices=[
            "xlm-roberta-base",          # 多语言RoBERTa（推荐，支持100+语言）
            "bert-base-multilingual-cased",  # 多语言BERT
            "distilbert-base-multilingual-cased",  # 轻量级多语言
        ],
        help="预训练模型名称"
    )

    # 训练参数
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="批次大小")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # 输出参数
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/final",
        help="模型输出目录"
    )

    # 设备参数
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备 (cuda/cpu/mps)，默认自动检测"
    )

    args = parser.parse_args()

    # 自动检测最佳设备
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    main(args)
