"""
Filtry jakosci kadrow twarzy.

Bardzo male kadry (np. 32x32) daja niewiarygodne embeddingi - lepiej je
odrzucic, bo psuja zarowno wizualizacje, jak i uczenie metryki.
"""
from __future__ import annotations

import numpy as np


def min_sides(faces) -> np.ndarray:
    if len(faces) == 0:
        return np.array([])
    return np.array([min(f.shape[0], f.shape[1]) for f in faces])


def size_mask(faces, min_size: int) -> np.ndarray:
    """Maska True/False: ktore kadry sa wystarczajaco duze (>= min_size px)."""
    if min_size <= 0 or len(faces) == 0:
        return np.ones(len(faces), dtype=bool)
    return min_sides(faces) >= min_size


def apply_mask(mask, names, X, faces, *extra):
    """Filtruje rownolegle names / X / faces (i dowolne dodatkowe listy)."""
    mask = np.asarray(mask, dtype=bool)
    names_f = [n for n, m in zip(names, mask) if m]
    faces_f = [f for f, m in zip(faces, mask) if m]
    X_f = X[mask] if len(X) else X
    extra_f = tuple([e for e, m in zip(lst, mask) if m] for lst in extra)
    return (names_f, X_f, faces_f, *extra_f)
