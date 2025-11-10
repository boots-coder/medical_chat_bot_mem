"""
测试脚本：测试上下文感知对话聚类器
"""
import json
from context_aware_clusterer import ContextAwareDialogueClusterer


def main():
    # 加载测试数据
    with open('test_dialogue_data.json', 'r', encoding='utf-8') as f:
        dialogue_list = json.load(f)

    print(f"总对话轮数: {len(dialogue_list)}\n")
    print("=" * 80)

    # 初始化聚类器
    clusterer = ContextAwareDialogueClusterer(
        min_cluster_size=3,
        min_samples=2
    )

    # 执行聚类
    result = clusterer.process(dialogue_list, return_details=True)

    labels = result['labels']
    representatives = result['representatives']
    cluster_dialogues = result['cluster_dialogues']
    details = result['details']

    # 2. 打印聚类统计信息
    print(f"\n聚类统计:")
    print(f"  - 有效簇数量: {details['n_clusters']}")
    print(f"  - 噪声点数量: {details['n_noise_points']}")
    print("=" * 80)

    # 3. 打印每个簇的代表性对话
    print(f"\n每个簇的代表性对话:\n")
    for cluster_id in sorted(representatives.keys()):
        if cluster_id == -1:
            print(f"簇 {cluster_id} (噪声点) - 代表性对话:")
        else:
            print(f"簇 {cluster_id} - 代表性对话:")

        for rep in representatives[cluster_id][:5]:  # 只显示前5个代表
            print(f"  [{rep['role']}] {rep['content']}")
        print()

    print("=" * 80)

    # 1. 打印每个簇的完整对话内容
    print(f"\n每个簇的完整对话内容:\n")
    for cluster_id in sorted(cluster_dialogues.keys()):
        dialogues = cluster_dialogues[cluster_id]

        if cluster_id == -1:
            print(f"簇 {cluster_id} (噪声点) - 共 {len(dialogues)} 轮对话:")
        else:
            print(f"簇 {cluster_id} - 共 {len(dialogues)} 轮对话:")

        for turn in dialogues:
            print(f"  [{turn['role']}] {turn['content']}")
        print()
        print("-" * 80)


if __name__ == "__main__":
    main()
