"""
Grupowanie twarzy w osoby (bez etykiet).

- cluster_dbscan          : DBSCAN na dystansie kosinusowym (sklearn).
- cluster_chinese_whispers: prosty graf podobienstwa + glosowanie sasiadow
                            (metoda polecana w dokumentacji face_recognition,
                            tu w czystym numpy).

Zwracaja wektor etykiet klastrow (int). W DBSCAN -1 = szum (nieprzypisane).
"""
from __future__ import annotations

import numpy as np


def cluster_dbscan(X: np.ndarray, eps: float = 0.35, min_samples: int = 2,
                   metric: str = "cosine") -> np.ndarray:
    from sklearn.cluster import DBSCAN

    return DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit_predict(X)


def cluster_chinese_whispers(X: np.ndarray, threshold: float = 0.6,
                             iterations: int = 20, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    sim = Xn @ Xn.T
    n = len(X)
    labels = np.arange(n)
    neighbors = [
        np.where((sim[i] >= threshold) & (np.arange(n) != i))[0] for i in range(n)
    ]
    for _ in range(iterations):
        for i in rng.permutation(n):
            if len(neighbors[i]) == 0:
                continue
            votes: dict[int, float] = {}
            for j in neighbors[i]:
                votes[labels[j]] = votes.get(labels[j], 0.0) + float(sim[i, j])
            labels[i] = max(votes, key=votes.get)
    _, labels = np.unique(labels, return_inverse=True)  # przenumeruj 0..k-1
    return labels
