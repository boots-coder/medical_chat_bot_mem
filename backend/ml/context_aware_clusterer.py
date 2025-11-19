import numpy as np
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from typing import List, Dict, Tuple


class ContextAwareDialogueClusterer:
    """Context-aware dialogue clusterer - Q&A pair level, fusing only previous context"""

    def __init__(
        self,
        sbert_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        current_weight: float = 0.5,
        min_cluster_size: int = 2,
        min_samples: int = 1
    ):
        """
        Args:
            sbert_model_name: SBERT model name
            current_weight: Weight of current Q&A pair (remaining weight distributed to history with quadratic decay)
            min_cluster_size: HDBSCAN minimum cluster size
            min_samples: HDBSCAN density estimation parameter
        """
        self.sbert = SentenceTransformer(sbert_model_name)
        self.current_weight = current_weight
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def _pair_dialogues(self, dialogue_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Pair user-assistant dialogues into Q&A pairs

        Args:
            dialogue_list: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

        Returns:
            qa_pairs: [{"user": "...", "assistant": "...", "combined": "User: ...\nAssistant: ..."}]
        """
        qa_pairs = []
        for i in range(0, len(dialogue_list), 2):
            if i + 1 < len(dialogue_list):
                user_content = dialogue_list[i]["content"]
                assistant_content = dialogue_list[i + 1]["content"]
                combined = f"User: {user_content}\nAssistant: {assistant_content}"
                qa_pairs.append({
                    "user": user_content,
                    "assistant": assistant_content,
                    "combined": combined
                })
        return qa_pairs

    def _compute_history_weights(self, n_history: int, total_weight: float) -> np.ndarray:
        """
        Compute history weights with quadratic decay

        Solve: w + w^2 + w^3 + ... + w^n_history = total_weight
        Use binary search for numerical solution

        Args:
            n_history: Number of historical Q&A pairs
            total_weight: Total weight allocated to history

        Returns:
            weights: array of shape (n_history,), from most recent to oldest: [w, w^2, w^3, ...]
        """
        if n_history == 0:
            return np.array([])

        if n_history == 1:
            return np.array([total_weight])

        # Binary search to solve for w
        def sum_geometric_powers(w, n):
            """Calculate w + w^2 + ... + w^n"""
            if abs(w - 1.0) < 1e-10:
                return n
            return w * (1 - w**n) / (1 - w)

        # Binary search
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
            w = mid  # Use the last mid value

        # Generate weight sequence [w, w^2, w^3, ...]
        weights = np.array([w ** (i + 1) for i in range(n_history)])

        # Normalize to ensure sum equals total_weight
        weights = weights / weights.sum() * total_weight

        return weights

    def encode_with_context(
        self,
        dialogue_list: List[Dict[str, str]]
    ) -> Tuple[np.ndarray, List[Dict[str, str]]]:
        """
        Encode each Q&A pair with context awareness (fusing only previous context, weights with quadratic decay)

        For Q&A pair[i], its vector = w_curr * vec[i] + Σ(w_history_j * vec[i-j])

        Args:
            dialogue_list: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

        Returns:
            embeddings: shape=(n_qa_pairs, embedding_dim)
            qa_pairs: List of paired Q&A pairs
        """
        # Pair dialogues
        qa_pairs = self._pair_dialogues(dialogue_list)
        n = len(qa_pairs)

        if n == 0:
            return np.array([]), []

        # Encode all Q&A pairs' combined text
        combined_texts = [qa["combined"] for qa in qa_pairs]
        all_embeddings = self.sbert.encode(combined_texts, convert_to_numpy=True)

        context_embeddings = []

        for i in range(n):
            emb_curr = all_embeddings[i]

            # First Q&A pair, give all weight to current
            if i == 0:
                context_emb = emb_curr
            else:
                # Compute history weights
                n_history = i
                history_weight = 1.0 - self.current_weight
                history_weights = self._compute_history_weights(n_history, history_weight)

                # Weighted fusion: current + history
                context_emb = self.current_weight * emb_curr

                # Accumulate history from most recent to oldest
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
        Execute HDBSCAN clustering

        Args:
            dialogue_list: Dialogue list

        Returns:
            labels: Cluster label for each Q&A pair, -1 indicates noise point
            clusterer: HDBSCAN clusterer object
            embeddings: Context-aware vectors
            qa_pairs: List of paired Q&A pairs
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
        Extract representative Q&A pairs for each cluster (closest K to cluster center)

        Args:
            qa_pairs: List of Q&A pairs
            embeddings: Context-aware vectors
            labels: Cluster labels
            n_representatives: Number of representatives to select per cluster

        Returns:
            representatives: {cluster_id: [{"index": i, "user": "...", "assistant": "...", "combined": "..."}]}
                             cluster_id=-1 indicates noise cluster
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
        Get complete Q&A pair content for each cluster (maintaining original order)

        Args:
            qa_pairs: List of Q&A pairs
            labels: Cluster labels

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
        Complete clustering pipeline

        Process:
        1. Pair user-assistant into Q&A pairs
        2. Context-aware encoding (fusing only previous context, weights with quadratic decay)
        3. HDBSCAN clustering (automatically determine cluster count, preserve noise points)
        4. Extract representative Q&A pairs for each cluster
        5. Organize complete Q&A pairs for each cluster

        Args:
            dialogue_list: Raw dialogue [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
            return_details: Whether to return detailed information

        Returns:
            {
                "labels": Cluster labels (corresponding to Q&A pairs),
                "qa_pairs": List of paired Q&A pairs,
                "representatives": Representative Q&A pairs,
                "cluster_dialogues": Complete Q&A pairs for each cluster,
                "details": {  # Only returned when return_details=True
                    "n_clusters": Number of valid clusters (excluding noise),
                    "n_noise_points": Number of noise points,
                    "embeddings": Context-aware vectors
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
