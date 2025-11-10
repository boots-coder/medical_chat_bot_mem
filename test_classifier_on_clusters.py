#!/usr/bin/env python3
"""
测试 LightweightMedicalClassifier 对聚类结果的分类效果
"""

from LightweightMedicalClassifier import LightweightMedicalClassifier


def main():
    # 初始化分类器
    classifier = LightweightMedicalClassifier()

    # 硬编码的聚类代表性对话
    cluster_texts = {
        -1: """你好医生
您好,请问有什么可以帮助您的?
有点恶心,光线强的时候眼睛不舒服
好的,谢谢医生。对了,今天天气好冷啊
糖尿病患者应该少吃高糖、高脂食物,多吃粗粮、蔬菜,每餐七八分饱""",

        0: """另外建议餐后2小时也测一下血糖,全面了解血糖控制情况
餐后血糖多少算正常?
餐后2小时血糖应该小于10.0mmol/L,最好控制在7.8mmol/L以下""",

        1: """建议每周至少运动3-5次,每次30分钟以上,可以选择快走、慢跑、游泳等
当然可以,建议餐后半小时到一小时运动,快走、太极拳都很适合
运动多长时间合适?
我现在体重还好,就是缺乏运动""",

        2: """嗯,有任何不适及时就医,不要拖延
好的,我知道了,不会再犯这个错误了
那怎么办?我现在很难受
如果持续剧痛或出现呕血、黑便,需要立即就医
如果出现突然剧烈头痛、意识改变、肢体无力、说话不清等症状,需要立即就医""",

        3: """建议减轻体重减少关节负担,可以适当补充氨糖,疼痛时外用止痛药膏
老年人适当补钙有益,建议每天钙摄入量1000-1200mg,可以通过饮食或补充剂
她需要补钙吗?
需要治疗吗?"""
    }

    print("=" * 80)
    print("LightweightMedicalClassifier 聚类分类测试")
    print("=" * 80)
    print()

    # 对每个簇的文本进行分类
    for cluster_id, text in cluster_texts.items():
        cluster_label = "噪声点" if cluster_id == -1 else f"簇 {cluster_id}"
        print(f"\n{'=' * 80}")
        print(f"测试 {cluster_label}")
        print(f"{'=' * 80}")
        print(f"\n原始文本:")
        print(f"{text[:200]}..." if len(text) > 200 else text)
        print(f"\n{'−' * 80}")

        # 进行分类
        result = classifier.classify(text)

        print(f"\n分类结果:")
        if result is None:
            print(f"  状态: ❌ 分类失败（API 调用或解析错误）")
        elif result is True:
            print(f"  状态: ✓ 医疗相关")
        else:  # result is False
            print(f"  状态: ✗ 非医疗相关")

        print()

    print("=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
