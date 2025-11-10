import numpy as np
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from typing import List, Dict, Tuple


class ContextAwareDialogueClusterer:
    """基于上下文感知的对话聚类器"""
    
    def __init__(
        self,
        sbert_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        context_weight_prev: float = 0.1,
        context_weight_curr: float = 0.6,
        context_weight_next: float = 0.3,
        min_cluster_size: int = 2,
        min_samples: int = 1
    ):
        """
        Args:
            sbert_model_name: SBERT模型名称
            context_weight_prev: 前一个turn的权重 (w_prev)
            context_weight_curr: 当前turn的权重 (w_curr)
            context_weight_next: 后一个turn的权重 (w_next)
            min_cluster_size: HDBSCAN最小簇大小
            min_samples: HDBSCAN密度估计参数
        """
        self.sbert = SentenceTransformer(sbert_model_name)
        self.w_prev = context_weight_prev
        self.w_curr = context_weight_curr
        self.w_next = context_weight_next
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
    
    def encode_with_context(
        self, 
        dialogue_list: List[Dict[str, str]]
    ) -> np.ndarray:
        """
        对每个turn进行上下文感知编码（加权平均方式）
        
        对于turn[i]，其向量 = w_prev * vec[i-1] + w_curr * vec[i] + w_next * vec[i+1]
        
        Args:
            dialogue_list: [{"role": "user", "content": "..."}, ...]
        
        Returns:
            embeddings: shape=(n_turns, embedding_dim)
        """
        n = len(dialogue_list)
        contents = [turn["content"] for turn in dialogue_list]
        
        all_embeddings = self.sbert.encode(contents, convert_to_numpy=True)
        
        context_embeddings = []
        for i in range(n):
            emb_curr = all_embeddings[i]
            emb_prev = all_embeddings[i-1] if i > 0 else np.zeros_like(emb_curr)
            emb_next = all_embeddings[i+1] if i < n-1 else np.zeros_like(emb_curr)
            
            if i == 0:
                w_p, w_c, w_n = 0, 0.7, 0.3
            elif i == n-1:
                w_p, w_c, w_n = 0.3, 0.7, 0
            else:
                w_p, w_c, w_n = self.w_prev, self.w_curr, self.w_next
            
            context_emb = w_p * emb_prev + w_c * emb_curr + w_n * emb_next
            context_embeddings.append(context_emb)
        
        return np.array(context_embeddings)
    
    def cluster_dialogue(
        self, 
        dialogue_list: List[Dict[str, str]]
    ) -> Tuple[np.ndarray, HDBSCAN]:
        """
        执行HDBSCAN聚类
        
        Args:
            dialogue_list: 对话列表
        
        Returns:
            labels: 每个turn的簇标签，-1表示噪声点
            clusterer: HDBSCAN聚类器对象
        """
        embeddings = self.encode_with_context(dialogue_list)
        
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        labels = clusterer.fit_predict(embeddings)
        return labels, clusterer
    
    def extract_representative_texts(
        self,
        dialogue_list: List[Dict[str, str]],
        embeddings: np.ndarray,
        labels: np.ndarray,
        n_representatives: int = 5
    ) -> Dict[int, List[Dict]]:
        """
        提取每个簇的代表性文本（距离簇中心最近的K个）
        
        Args:
            dialogue_list: 原始对话
            embeddings: 上下文感知向量
            labels: 聚类标签
            n_representatives: 每个簇选择的代表数量
        
        Returns:
            representatives: {cluster_id: [{"index": i, "content": "...", "role": "..."}]}
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
                        "content": dialogue_list[idx]["content"],
                        "role": dialogue_list[idx]["role"]
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
                        "content": dialogue_list[cluster_indices[idx]]["content"],
                        "role": dialogue_list[cluster_indices[idx]]["role"]
                    }
                    for idx in top_n_idx
                ]
        
        return representatives
    
    def get_cluster_dialogues(
        self,
        dialogue_list: List[Dict[str, str]],
        labels: np.ndarray
    ) -> Dict[int, List[Dict[str, str]]]:
        """
        获取每个簇的完整对话内容（保持原始顺序）
        
        Args:
            dialogue_list: 原始对话
            labels: 聚类标签
        
        Returns:
            cluster_dialogues: {cluster_id: [dialogue_turns]}
        """
        cluster_dialogues = {}
        unique_labels = set(labels)
        
        for label in unique_labels:
            cluster_dialogues[label] = []
        
        for i, turn in enumerate(dialogue_list):
            label = labels[i]
            cluster_dialogues[label].append(turn)
        
        return cluster_dialogues
    
    def process(
        self,
        dialogue_list: List[Dict[str, str]],
        return_details: bool = False
    ) -> Dict:
        """
        完整聚类流程
        
        流程：
        1. 上下文感知编码（加权平均）
        2. HDBSCAN聚类（自动确定簇数，保留噪声点）
        3. 提取每个簇的代表文本
        4. 组织每个簇的完整对话
        
        Args:
            dialogue_list: 原始对话
            return_details: 是否返回详细信息
        
        Returns:
            {
                "labels": 聚类标签,
                "representatives": 代表文本,
                "cluster_dialogues": 每个簇的完整对话,
                "details": {  # 仅当return_details=True时返回
                    "n_clusters": 有效簇数量（不含噪声）,
                    "n_noise_points": 噪声点数量,
                    "embeddings": 上下文感知向量
                }
            }
        """
        embeddings = self.encode_with_context(dialogue_list)
        
        labels, clusterer = self.cluster_dialogue(dialogue_list)
        
        representatives = self.extract_representative_texts(
            dialogue_list, embeddings, labels
        )
        
        cluster_dialogues = self.get_cluster_dialogues(
            dialogue_list, labels
        )
        
        result = {
            "labels": labels.tolist(),
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