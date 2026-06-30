"""
Filtry jakosci i przynaleznosci kadrow.

- size_mask        : odrzuca zbyt male kadry (np. 32x32 -> niewiarygodne embeddingi).
- folder_keep_mask : odrzuca twarze z folderow-nie-osob (zwierzeta, nierozpoznane);
                     opcjonalnie zostawia tylko sklasyfikowane osoby.
- apply_mask       : filtruje rownolegle names / X / faces.
"""
from __future__ import annotations

import numpy as np


def min_sides(faces) -> np.ndarray:
    if len(faces) == 0:
        return np.array([])
    return np.array([min(f.shape[0], f.shape[1]) for f in faces])


def size_mask(faces, min_size: int) -> np.ndarray:
    if min_size <= 0 or len(faces) == 0:
        return np.ones(len(faces), dtype=bool)
    return min_sides(faces) >= min_size


def folder_keep_mask(folders, ignore, only_labeled: bool = False) -> np.ndarray:
    """True = zostaw. Wyrzuca foldery z ignore; gdy only_labeled - takze 'luzem' (None)."""
    ignore = {str(s).lower() for s in ignore}
    keep = []
    for f in folders:
        if f is not None and f.strip().lower() in ignore:
            keep.append(False)
        elif only_labeled and f is None:
            keep.append(False)
        else:
            keep.append(True)
    return np.array(keep, dtype=bool)


def apply_mask(mask, names, X, faces, *extra):
    """Filtruje rownolegle names / X / faces (i dowolne dodatkowe listy)."""
    mask = np.asarray(mask, dtype=bool)
    names_f = [n for n, m in zip(names, mask) if m]
    faces_f = [f for f, m in zip(faces, mask) if m]
    X_f = X[mask] if len(X) else X
    extra_f = tuple([e for e, m in zip(lst, mask) if m] for lst in extra)
    return (names_f, X_f, faces_f, *extra_f)
