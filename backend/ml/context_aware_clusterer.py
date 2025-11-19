import numpy as np
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from typing import List, Dict, Tuple


class ContextAwareDialogueClusterer:
    """基于上下文感知的对话聚类器 - 问答对级别，只融合上文"""

    def __init__(
        self,
        sbert_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        current_weight: float = 0.5,
        min_cluster_size: int = 2,
        min_samples: int = 1
    ):
        """
        Args:
            sbert_model_name: SBERT模型名称
            current_weight: 当前问答对的权重 (剩余权重按平方递减分配给历史)
            min_cluster_size: HDBSCAN最小簇大小
            min_samples: HDBSCAN密度估计参数
        """
        self.sbert = SentenceTransformer(sbert_model_name)
        self.current_weight = current_weight
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def _pair_dialogues(self, dialogue_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        将user-assistant对话配对成问答对

        Args:
            dialogue_list: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

        Returns:
            qa_pairs: [{"user": "...", "assistant": "...", "combined": "用户：...\n助手：..."}]
        """
        qa_pairs = []
        for i in range(0, len(dialogue_list), 2):
            if i + 1 < len(dialogue_list):
                user_content = dialogue_list[i]["content"]
                assistant_content = dialogue_list[i + 1]["content"]
                combined = f"用户：{user_content}\n助手：{assistant_content}"
                qa_pairs.append({
                    "user": user_content,
                    "assistant": assistant_content,
                    "combined": combined
                })
        return qa_pairs

    def _compute_history_weights(self, n_history: int, total_weight: float) -> np.ndarray:
        """
        计算历史权重，按平方递减分配

        求解: w + w^2 + w^3 + ... + w^n_history = total_weight
        使用二分法数值求解

        Args:
            n_history: 历史问答对数量
            total_weight: 分配给历史的总权重

        Returns:
            weights: array of shape (n_history,)，从最近到最远: [w, w^2, w^3, ...]
        """
        if n_history == 0:
            return np.array([])

        if n_history == 1:
            return np.array([total_weight])

        # 二分法求解 w
        def sum_geometric_powers(w, n):
            """计算 w + w^2 + ... + w^n"""
            if abs(w - 1.0) < 1e-10:
                return n
            return w * (1 - w**n) / (1 - w)

        # 二分搜索
        left, right = 0.0, 1.0
        epsilon = 1e-8
        max_iter = 100

        for _ in range(max_iter):
            mid = (left + right) / 2
            sum_val = sum_geometric_powers(mid, n_history)

            if abs(sum_val - total_weight) < epsilon:
                w = mid
                break

            if sum_val < total_weight:
                left = mid
            else:
                right = mid
        else:
            w = mid  # 使用最后的mid值

        # 生成权重序列 [w, w^2, w^3, ...]
        weights = np.array([w ** (i + 1) for i in range(n_history)])

        # 归一化确保总和等于 total_weight
        weights = weights / weights.sum() * total_weight

        return weights

    def encode_with_context(
        self,
        dialogue_list: List[Dict[str, str]]
    ) -> Tuple[np.ndarray, List[Dict[str, str]]]:
        """
        对每个问答对进行上下文感知编码（只融合上文，权重平方递减）

        对于问答对[i]，其向量 = w_curr * vec[i] + Σ(w_history_j * vec[i-j])

        Args:
            dialogue_list: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

        Returns:
            embeddings: shape=(n_qa_pairs, embedding_dim)
            qa_pairs: 配对后的问答对列表
        """
        # 配对对话
        qa_pairs = self._pair_dialogues(dialogue_list)
        n = len(qa_pairs)

        if n == 0:
            return np.array([]), []

        # 编码所有问答对的combined文本
        combined_texts = [qa["combined"] for qa in qa_pairs]
        all_embeddings = self.sbert.encode(combined_texts, convert_to_numpy=True)

        context_embeddings = []

        for i in range(n):
            emb_curr = all_embeddings[i]

            # 第一个问答对，权重全给当前
            if i == 0:
                context_emb = emb_curr
            else:
                # 计算历史权重
                n_history = i
                history_weight = 1.0 - self.current_weight
                history_weights = self._compute_history_weights(n_history, history_weight)

                # 加权融合：当前 + 历史
                context_emb = self.current_weight * emb_curr

                # 从最近到最远累加历史
                for j in range(n_history):
                    hist_idx = i - j - 1  # i-1, i-2, i-3, ...
                    context_emb += history_weights[j] * all_embeddings[hist_idx]

            context_embeddings.append(context_emb)

        return np.array(context_embeddings), qa_pairs
    
    def cluster_dialogue(
        self,
        dialogue_list: List[Dict[str, str]]
    ) -> Tuple[np.ndarray, HDBSCAN, np.ndarray, List[Dict[str, str]]]:
        """
        执行HDBSCAN聚类

        Args:
            dialogue_list: 对话列表

        Returns:
            labels: 每个问答对的簇标签，-1表示噪声点
            clusterer: HDBSCAN聚类器对象
            embeddings: 上下文感知向量
            qa_pairs: 配对后的问答对列表
        """
        embeddings, qa_pairs = self.encode_with_context(dialogue_list)

        if len(embeddings) == 0:
            return np.array([]), None, embeddings, qa_pairs

        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )

        labels = clusterer.fit_predict(embeddings)
        return labels, clusterer, embeddings, qa_pairs
    
    def extract_representative_texts(
        self,
        qa_pairs: List[Dict[str, str]],
        embeddings: np.ndarray,
        labels: np.ndarray,
        n_representatives: int = 5
    ) -> Dict[int, List[Dict]]:
        """
        提取每个簇的代表性问答对（距离簇中心最近的K个）

        Args:
            qa_pairs: 问答对列表
            embeddings: 上下文感知向量
            labels: 聚类标签
            n_representatives: 每个簇选择的代表数量

        Returns:
            representatives: {cluster_id: [{"index": i, "user": "...", "assistant": "...", "combined": "..."}]}
                             cluster_id=-1 表示噪声簇
        """
        representatives = {}
        unique_labels = set(labels)

        for label in unique_labels:
            cluster_mask = labels == label
            cluster_embeddings = embeddings[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]

            if label == -1:
                representatives[-1] = [
                    {
                        "index": int(idx),
                        "user": qa_pairs[idx]["user"],
                        "assistant": qa_pairs[idx]["assistant"],
                        "combined": qa_pairs[idx]["combined"]
                    }
                    for idx in cluster_indices
                ]
            else:
                centroid = cluster_embeddings.mean(axis=0)
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                top_n_idx = np.argsort(distances)[:n_representatives]

                representatives[int(label)] = [
                    {
                        "index": int(cluster_indices[idx]),
                        "user": qa_pairs[cluster_indices[idx]]["user"],
                        "assistant": qa_pairs[cluster_indices[idx]]["assistant"],
                        "combined": qa_pairs[cluster_indices[idx]]["combined"]
                    }
                    for idx in top_n_idx
                ]

        return representatives
    
    def get_cluster_dialogues(
        self,
        qa_pairs: List[Dict[str, str]],
        labels: np.ndarray
    ) -> Dict[int, List[Dict[str, str]]]:
        """
        获取每个簇的完整问答对内容（保持原始顺序）

        Args:
            qa_pairs: 问答对列表
            labels: 聚类标签

        Returns:
            cluster_dialogues: {cluster_id: [qa_pairs]}
        """
        cluster_dialogues = {}
        unique_labels = set(labels)

        for label in unique_labels:
            cluster_dialogues[int(label)] = []

        for i, qa_pair in enumerate(qa_pairs):
            label = int(labels[i])
            cluster_dialogues[label].append(qa_pair)

        return cluster_dialogues
    
    def process(
        self,
        dialogue_list: List[Dict[str, str]],
        return_details: bool = False
    ) -> Dict:
        """
        完整聚类流程

        流程：
        1. 配对user-assistant为问答对
        2. 上下文感知编码（只融合上文，权重平方递减）
        3. HDBSCAN聚类（自动确定簇数，保留噪声点）
        4. 提取每个簇的代表问答对
        5. 组织每个簇的完整问答对

        Args:
            dialogue_list: 原始对话 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
            return_details: 是否返回详细信息

        Returns:
            {
                "labels": 聚类标签 (对应问答对),
                "qa_pairs": 配对后的问答对列表,
                "representatives": 代表问答对,
                "cluster_dialogues": 每个簇的完整问答对,
                "details": {  # 仅当return_details=True时返回
                    "n_clusters": 有效簇数量（不含噪声）,
                    "n_noise_points": 噪声点数量,
                    "embeddings": 上下文感知向量
                }
            }
        """
        labels, clusterer, embeddings, qa_pairs = self.cluster_dialogue(dialogue_list)

        if len(qa_pairs) == 0:
            return {
                "labels": [],
                "qa_pairs": [],
                "representatives": {},
                "cluster_dialogues": {}
            }

        representatives = self.extract_representative_texts(
            qa_pairs, embeddings, labels
        )

        cluster_dialogues = self.get_cluster_dialogues(
            qa_pairs, labels
        )

        result = {
            "labels": labels.tolist(),
            "qa_pairs": qa_pairs,
            "representatives": representatives,
            "cluster_dialogues": cluster_dialogues
        }

        if return_details:
            n_noise = np.sum(labels == -1)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            result["details"] = {
                "n_clusters": n_clusters,
                "n_noise_points": int(n_noise),
                "embeddings": embeddings
            }

        return result