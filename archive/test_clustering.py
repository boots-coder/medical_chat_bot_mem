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
    qa_pairs = result['qa_pairs']
    representatives = result['representatives']
    cluster_dialogues = result['cluster_dialogues']
    details = result['details']

    print(f"\n配对后的问答对数量: {len(qa_pairs)}")

    # 2. 打印聚类统计信息
    print(f"\n聚类统计:")
    print(f"  - 有效簇数量: {details['n_clusters']}")
    print(f"  - 噪声点数量: {details['n_noise_points']}")
    print("=" * 80)

    # 3. 打印每个簇的代表性问答对
    print(f"\n每个簇的代表性问答对:\n")
    for cluster_id in sorted(representatives.keys()):
        if cluster_id == -1:
            print(f"簇 {cluster_id} (噪声点) - 代表性问答对:")
        else:
            print(f"簇 {cluster_id} - 代表性问答对:")

        for rep in representatives[cluster_id][:5]:  # 只显示前5个代表
            print(f"  [问答对 #{rep['index']}]")
            print(f"    用户: {rep['user']}")
            print(f"    助手: {rep['assistant']}")
            print()

    print("=" * 80)

    # 4. 打印每个簇的完整问答对内容
    print(f"\n每个簇的完整问答对内容:\n")
    for cluster_id in sorted(cluster_dialogues.keys()):
        qa_list = cluster_dialogues[cluster_id]

        if cluster_id == -1:
            print(f"簇 {cluster_id} (噪声点) - 共 {len(qa_list)} 个问答对:")
        else:
            print(f"簇 {cluster_id} - 共 {len(qa_list)} 个问答对:")

        for qa in qa_list:
            print(f"  用户: {qa['user']}")
            print(f"  助手: {qa['assistant']}")
            print()
        print("-" * 80)


if __name__ == "__main__":
    main()
