"""
Uczenie metryki z nadzorem (supervised metric learning).

Majac etykiety kto-jest-kim, uczymy przeksztalcenia embeddingow tak, by TA SAMA
osoba byla blizej, a ROZNE osoby dalej. Naprawia to sytuacje, gdy nienadzorowany
uklad (UMAP/PCA) stawia ta sama osobe daleko, a dwie rozne blisko.

Metody:
    lda  - Linear Discriminant Analysis (scikit-learn)
    nca  - Neighbourhood Components Analysis (scikit-learn)
    lmnn - Large Margin Nearest Neighbor (metric-learn; pip install metric-learn)

Transformacja uczy sie na twarzach z etykieta i jest stosowana do wszystkich.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def reduce_or_pad(Y, n: int = 3):
    """Sprowadza wynik do n wymiarow: PCA gdy za duzo, zera gdy za malo."""
    Y = np.asarray(Y, dtype=float)
    if Y.shape[1] == n:
        return Y
    if Y.shape[1] > n:
        from sklearn.decomposition import PCA
        return PCA(n_components=n).fit_transform(Y)
    pad = np.zeros((len(Y), n - Y.shape[1]))
    return np.hstack([Y, pad])


def _labeled_subset(X, labels):
    mask = np.array([l is not None for l in labels])
    if mask.sum() < 2:
        raise SystemExit("Potrzeba etykiet dla co najmniej 2 twarzy.")
    yl = [l for l, m in zip(labels, mask) if m]
    if len(set(yl)) < 2:
        raise SystemExit("Potrzeba co najmniej 2 roznych osob z etykietami.")
    return mask, X[mask], yl


def learn_metric(X, labels, method: str = "lmnn", n_components: int = 3):
    """Uczy transformacji na oetykietowanych twarzach i stosuje ja do wszystkich.
    Zwraca wspolrzedne (N x n_components)."""
    method = method.lower()
    _mask, Xl, yl = _labeled_subset(X, labels)

    if method == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        nc = min(n_components, len(set(yl)) - 1)
        model = LinearDiscriminantAnalysis(n_components=nc).fit(Xl, yl)
        return reduce_or_pad(model.transform(X), n_components)

    if method == "nca":
        from sklearn.neighbors import NeighborhoodComponentsAnalysis
        nc = min(n_components, X.shape[1])
        model = NeighborhoodComponentsAnalysis(n_components=nc, random_state=42).fit(Xl, yl)
        return reduce_or_pad(model.transform(X), n_components)

    if method == "lmnn":
        try:
            from metric_learn import LMNN
        except ImportError as exc:
            raise SystemExit("LMNN wymaga: pip install metric-learn") from exc
        min_class = min(Counter(yl).values())
        k = max(1, min(3, min_class - 1))
        try:                                   # nowsze metric-learn
            model = LMNN(n_neighbors=k, n_components=n_components)
        except TypeError:                      # starsze API
            model = LMNN(k=k)
        model.fit(Xl, yl)
        return reduce_or_pad(model.transform(X), n_components)

    raise ValueError(f"Nieznana metoda metryki: {method} (dostepne: lda, nca, lmnn)")
