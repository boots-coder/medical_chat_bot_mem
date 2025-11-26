"""
RAG意图分类器测试脚本
加载训练好的模型进行推理测试
"""
import json
import argparse
from pathlib import Path
from training.local_classifier import LocalRAGIntentClassifier


def test_interactive(classifier):
    """交互式测试模式"""
    print("\n" + "="*60)
    print("交互式测试模式")
    print("输入 'quit' 或 'exit' 退出")
    print("="*60 + "\n")

    test_cases = [
        {
            "query": "Is this headache worse than last time?",
            "context": "",
            "expected": True
        },
        {
            "query": "我今天头很痛",
            "context": "",
            "expected": False
        },
        {
            "query": "上次开的降压药我还能继续吃吗?",
            "context": "",
            "expected": True
        },
        {
            "query": "What should I do about my headache?",
            "context": "",
            "expected": False
        }
    ]

    print("【预设测试样例】\n")
    for i, case in enumerate(test_cases, 1):
        result = classifier.predict(
            query=case['query'],
            short_term_context=case['context']
        )

        status = "✓" if result['need_rag'] == case['expected'] else "✗"
        print(f"{status} 样例 {i}:")
        print(f"  Query: {case['query']}")
        print(f"  预测: need_rag={result['need_rag']}, confidence={result['confidence']:.3f}")
        print(f"  期望: need_rag={case['expected']}")
        print()

    print("\n【自定义测试】\n")
    while True:
        query = input("输入查询 (或 quit 退出): ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break

        if not query:
            continue

        context = input("输入上下文 (可选，直接回车跳过): ").strip()

        result = classifier.predict(
            query=query,
            short_term_context=context
        )

        print(f"\n预测结果:")
        print(f"  need_rag: {result['need_rag']}")
        print(f"  confidence: {result['confidence']:.3f}")
        print()


def test_batch(classifier, test_file):
    """批量测试模式"""
    print("\n" + "="*60)
    print(f"批量测试: {test_file}")
    print("="*60)

    with open(test_file, 'r', encoding='utf-8') as f:
        test_samples = json.load(f)

    print(f"\n测试样本数: {len(test_samples)}")

    correct = 0
    total = len(test_samples)

    results = []

    for sample in test_samples:
        result = classifier.predict(
            query=sample['query'],
            short_term_context=sample.get('short_term_context', '')
        )

        is_correct = result['need_rag'] == sample['need_rag']
        if is_correct:
            correct += 1

        results.append({
            'query': sample['query'],
            'expected': sample['need_rag'],
            'predicted': result['need_rag'],
            'confidence': result['confidence'],
            'correct': is_correct
        })

    accuracy = correct / total

    print(f"\n测试结果:")
    print(f"  准确率: {accuracy:.4f} ({correct}/{total})")

    # 显示错误样本
    errors = [r for r in results if not r['correct']]
    if errors:
        print(f"\n错误样本 ({len(errors)}个):")
        for i, error in enumerate(errors[:10], 1):  # 只显示前10个
            print(f"\n  {i}. Query: {error['query'][:60]}...")
            print(f"     期望: {error['expected']}, 预测: {error['predicted']}, 置信度: {error['confidence']:.3f}")

    return accuracy, results


def main(args):
    print("="*60)
    print("RAG意图分类器测试")
    print("="*60)
    print(f"模型路径: {args.model_dir}")
    print("="*60)

    # 加载模型
    print("\n加载模型...")
    classifier = LocalRAGIntentClassifier.load_model(
        model_dir=args.model_dir,
        device=args.device
    )

    if args.test_file:
        # 批量测试
        accuracy, results = test_batch(classifier, args.test_file)

        # 保存结果
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'accuracy': accuracy,
                    'results': results
                }, f, ensure_ascii=False, indent=2)

            print(f"\n✓ 测试结果已保存: {output_path}")

    if args.interactive or not args.test_file:
        # 交互式测试
        test_interactive(classifier)

    print("\n测试完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试RAG意图分类器")

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="训练好的模型目录"
    )

    parser.add_argument(
        "--test-file",
        type=str,
        default=None,
        help="测试数据文件 (JSON格式，可选)"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="启用交互式测试模式"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="测试结果输出文件"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备 (cuda/cpu/mps)，默认自动检测"
    )

    args = parser.parse_args()

    # 自动检测设备
    if args.device is None:
        import torch
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    main(args)
