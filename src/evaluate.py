"""
Pomiar jakosci rozdzielania osob: czy ta sama osoba jest blisko, a rozne daleko.

Liczy dla surowych embeddingow oraz dla kazdej metody uczenia metryki:
    recall@1   - odsetek twarzy, ktorych najblizszy sasiad to TA SAMA osoba
                 (1.0 = idealnie; rosnie, gdy rozne osoby przestaja byc najblizej)
    silhouette - jak ciasne i odseparowane sa osoby (-1..1, wyzej = lepiej)

Odporne na realny zbior:
    - foldery-nie-osoby (nierozpoznane, zwierzeta...) sa pomijane,
    - male kadry (<--min-face-size) sa odrzucane,
    - --drop-suspected pomija prawdopodobne pomylki w etykietach.

Przyklad:
    python evaluate.py
    python evaluate.py --min-face-size 64 --drop-suspected
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dataset import load_embeddings, load_labels, load_labels_from_folders, DEFAULT_IGNORE
from postprocess import l2_normalize
from metric_learning import learn_metric
from quality import size_mask, apply_mask
from labels_audit import drop_suspected


def recall_at_1(X, labels) -> float:
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(X)
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(axis=1)
    return float(np.mean([labels[i] == labels[nn[i]] for i in range(len(X))]))


def silhouette(X, labels) -> float:
    from sklearn.metrics import silhouette_score
    if len(set(labels)) < 2:
        return float("nan")
    return float(silhouette_score(X, labels))


def find_project_root(start: Path) -> Path:
    for c in (start, start.parent):
        if (c / "emb").exists():
            return c
    return start


ROOT = find_project_root(Path(__file__).resolve().parent)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ocena rozdzielania osob (recall@1, silhouette)")
    ap.add_argument("--emb", default=str(ROOT / "emb"))
    ap.add_argument("--faces", default=str(ROOT / "emb_images"))
    ap.add_argument("--labels", default="", help="CSV; domyslnie podfoldery emb_images")
    ap.add_argument("--methods", nargs="+", default=["lda", "nca", "lmnn"])
    ap.add_argument("--min-face-size", type=int, default=50)
    ap.add_argument("--ignore-folders", nargs="*", default=None)
    ap.add_argument("--drop-suspected", action="store_true",
                    help="Pomin prawdopodobne pomylki w etykietach")
    args = ap.parse_args()

    emb_dir, faces_dir = Path(args.emb), Path(args.faces)
    ignore = DEFAULT_IGNORE if args.ignore_folders is None else {s.lower() for s in args.ignore_folders}

    names, X, faces = load_embeddings(emb_dir, faces_dir)
    if len(names) == 0:
        print("Brak embeddingow.")
        return
    before = len(names)
    names, X, faces = apply_mask(size_mask(faces, args.min_face_size), names, X, faces)
    X = l2_normalize(X)

    labels = (load_labels(args.labels, names) if args.labels
              else load_labels_from_folders(faces_dir, names, ignore_folders=ignore))
    if args.drop_suspected:
        labels, n_susp = drop_suspected(labels, X)
        print(f"Pominieto {n_susp} podejrzanych etykiet.")

    mask = np.array([l is not None for l in labels])
    ye = [l for l, m in zip(labels, mask) if m]
    if len(set(ye)) < 2:
        print("Potrzeba >=2 osob z etykietami. Ulóz twarze w podfoldery emb_images/osoba_X/.")
        return
    Xe = X[mask]

    print(f"Twarzy: {before} -> po filtrze rozmiaru {len(names)} | oznaczonych: {len(ye)} | osob: {len(set(ye))}\n")
    print(f"{'przestrzen':18s} {'recall@1':>9s} {'silhouette':>11s}")
    print("-" * 40)
    print(f"{'surowe (raw)':18s} {recall_at_1(Xe, ye):9.3f} {silhouette(Xe, ye):11.3f}")
    for m in args.methods:
        try:
            Xt = learn_metric(X, labels, method=m, n_components=3)[mask]
            print(f"{m:18s} {recall_at_1(Xt, ye):9.3f} {silhouette(Xt, ye):11.3f}")
        except SystemExit as exc:
            print(f"{m:18s}  pominieto: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"{m:18s}  blad: {exc}")


if __name__ == "__main__":
    main()
