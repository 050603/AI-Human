from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from classroom_ai.uncertainty.entropy import shannon_entropy


@dataclass(frozen=True)
class SemanticCluster:
    cluster_id: int
    count: int
    representative_text: str
    member_indices: list[int]


class PairwiseEntailmentJudge(Protocol):
    def equivalent(self, left: str, right: str) -> bool: ...


class LexicalEntailmentJudge:
    """Offline fallback judge for tests; replace with NLI/API judge in production."""

    def equivalent(self, left: str, right: str) -> bool:
        return _normalize_text(left) == _normalize_text(right)


def compute_semantic_entropy(
    texts: list[str],
    judge: PairwiseEntailmentJudge | None = None,
) -> tuple[float, list[int], list[SemanticCluster]]:
    if not texts:
        return 0.0, [], []

    judge = judge or LexicalEntailmentJudge()
    edges = _build_mutual_entailment_edges(texts, judge)
    labels = _connected_component_labels(len(texts), edges)
    entropy = shannon_entropy(labels)

    clusters = []
    for cluster_id in sorted(set(labels)):
        members = [idx for idx, label in enumerate(labels) if label == cluster_id]
        clusters.append(SemanticCluster(cluster_id, len(members), texts[members[0]], members))
    return entropy, labels, clusters


def _build_mutual_entailment_edges(texts: list[str], judge: PairwiseEntailmentJudge) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if judge.equivalent(texts[i], texts[j]) and judge.equivalent(texts[j], texts[i]):
                edges.append((i, j))
    return edges


def _connected_component_labels(size: int, edges: list[tuple[int, int]]) -> list[int]:
    try:
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(range(size))
        graph.add_edges_from(edges)
        labels = [0] * size
        for component_id, nodes in enumerate(nx.connected_components(graph)):
            for node in nodes:
                labels[node] = component_id
        return labels
    except Exception:
        neighbors = {i: set() for i in range(size)}
        for left, right in edges:
            neighbors[left].add(right)
            neighbors[right].add(left)
        labels = [-1] * size
        component_id = 0
        for start in range(size):
            if labels[start] != -1:
                continue
            stack = [start]
            labels[start] = component_id
            while stack:
                node = stack.pop()
                for nxt in neighbors[node]:
                    if labels[nxt] == -1:
                        labels[nxt] = component_id
                        stack.append(nxt)
            component_id += 1
        return labels


def _normalize_text(text: str) -> str:
    return "".join(ch for ch in text.lower().strip() if ch.isalnum())
