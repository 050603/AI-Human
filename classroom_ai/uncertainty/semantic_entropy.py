from __future__ import annotations

import math
from dataclasses import dataclass

from classroom_ai.embedding.base import BaseEmbedder, Vector
from classroom_ai.uncertainty.entropy import shannon_entropy


@dataclass(frozen=True)
class SemanticCluster:
    cluster_id: int
    count: int
    representative_text: str
    member_indices: list[int]


def compute_semantic_entropy(
    texts: list[str],
    embedder: BaseEmbedder,
    similarity_threshold: float = 0.82,
) -> tuple[float, list[int], list[SemanticCluster]]:
    """Cluster semantically similar outputs and compute cluster-label entropy.

    This intentionally uses a small greedy cosine clustering implementation so
    the notebook prototype works in offline environments without scikit-learn.
    """

    if not texts:
        return 0.0, [], []
    embeddings = embedder.encode(texts, normalize=True)
    labels = _greedy_cosine_clusters(embeddings, similarity_threshold)
    entropy = shannon_entropy(labels)
    clusters = []
    for cluster_id in sorted(set(labels)):
        indices = [index for index, label in enumerate(labels) if label == cluster_id]
        clusters.append(
            SemanticCluster(
                cluster_id=cluster_id,
                count=len(indices),
                representative_text=texts[indices[0]],
                member_indices=indices,
            )
        )
    return entropy, labels, clusters


def _greedy_cosine_clusters(embeddings: list[Vector], similarity_threshold: float) -> list[int]:
    centroids: list[Vector] = []
    members: list[list[int]] = []
    labels: list[int] = []

    for index, vector in enumerate(embeddings):
        if not centroids:
            centroids.append(vector.copy())
            members.append([index])
            labels.append(0)
            continue

        similarities = [_dot(vector, centroid) for centroid in centroids]
        best_cluster = max(range(len(similarities)), key=similarities.__getitem__)
        if similarities[best_cluster] >= similarity_threshold:
            labels.append(best_cluster)
            members[best_cluster].append(index)
            centroids[best_cluster] = _normalize(_mean([embeddings[item] for item in members[best_cluster]]))
        else:
            labels.append(len(centroids))
            centroids.append(vector.copy())
            members.append([index])

    return labels


def _dot(left: Vector, right: Vector) -> float:
    return sum(a * b for a, b in zip(left, right))


def _mean(vectors: list[Vector]) -> Vector:
    if not vectors:
        return []
    width = len(vectors[0])
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(width)]


def _normalize(vector: Vector) -> Vector:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm > 0 else vector
