"""
Operacje na macierzy embeddingow (N x D).

- l2_normalize : skaluje kazdy wektor do dlugosci 1 (-> dystans kosinusowy).
- regress_out  : tlumi wplyw swiatla. Usuwa z embeddingow zmiennosc napedzana
                 kowiata (np. jasnoscia kadru), z regulowana sila (0..1).
- lda_reduce   : nadzorowana redukcja do 3D. Maksymalizuje roznice MIEDZY
                 osobami i minimalizuje zmiennosc WEWNATRZ osoby (a swiatlo
                 to wlasnie zmiennosc wewnatrz osoby). Wymaga etykiet.
"""
from __future__ import annotations

import numpy as np


def l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def regress_out(X: np.ndarray, covariate: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Tlumi w embeddingach zmiennosc napedzana kowiata (np. jasnosc kadru).

    strength = ile tej zmiennosci odjac (0..1):
        0.0 -> nic (surowe embeddingi),
        1.0 -> pelne usuniecie liniowego wplywu swiatla,
        wartosci posrednie -> twarz wazy odpowiednio mocniej niz swiatlo.

    Srednia jest zachowana (centrujemy kowiata), wiec skala embeddingow zostaje.
    NaN w kowiacie -> zwraca X bez zmian.
    """
    if strength <= 0:
        return X
    c = np.asarray(covariate, dtype=float)
    if np.isnan(c).any():
        print("regress_out: brak pelnych danych o jasnosci (NaN) - pomijam tlumienie swiatla.")
        return X
    c_centered = c - c.mean()
    denom = float(c_centered @ c_centered)
    if denom == 0:
        return X
    slope = (c_centered @ (X - X.mean(axis=0, keepdims=True))) / denom  # (D,)
    light_component = np.outer(c_centered, slope)                       # (N, D)
    return X - strength * light_component


def lda_reduce(X: np.ndarray, labels: list, n_components: int = 3) -> np.ndarray:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    classes = sorted(set(labels))
    usable = min(n_components, max(1, len(classes) - 1))
    lda = LinearDiscriminantAnalysis(n_components=usable)
    Y = lda.fit_transform(X, labels)
    if Y.shape[1] < n_components:  # dopelnij zerami do 3D (do rysowania)
        pad = np.zeros((Y.shape[0], n_components - Y.shape[1]))
        Y = np.hstack([Y, pad])
    return Y
