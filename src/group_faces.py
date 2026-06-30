"""
Grupowanie twarzy w osoby + generowanie etykiet.

Wczytuje embeddingi, (opcjonalnie) normalizuje i tlumi swiatlo, a nastepnie
grupuje je w osoby. Wynik zapisuje do emb/labels_auto.csv (kolumny: name,person),
ktory mozesz potem podac do wizualizacji (--reduce LDA / --color-by label).
labels_auto.csv mozesz tez recznie poprawic - to dobra baza pod prawdziwe etykiety.

Domyslnie twarz wazy mocniej niz swiatlo (L2-normalizacja + tlumienie swiatla).

Przyklady:
    python group_faces.py
    python group_faces.py --method chinese_whispers --threshold 0.6
    python group_faces.py --light-strength 0.5
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from dataset import load_embeddings, load_brightness
from postprocess import l2_normalize, regress_out
from cluster import cluster_dbscan, cluster_chinese_whispers


def find_project_root(start: Path) -> Path:
    for candidate in (start, start.parent):
        if (candidate / "emb").exists():
            return candidate
    return start


ROOT = find_project_root(Path(__file__).resolve().parent)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grupowanie twarzy -> labels_auto.csv")
    parser.add_argument("--method", choices=["dbscan", "chinese_whispers"], default="dbscan")
    parser.add_argument("--emb", default=str(ROOT / "emb"))
    parser.add_argument("--faces", default=str(ROOT / "emb_images"))
    parser.add_argument("--out", default=str(ROOT / "emb" / "labels_auto.csv"))
    parser.add_argument("--no-normalize", action="store_true", help="Wylacz L2-normalizacje")
    parser.add_argument("--light-strength", type=float, default=1.0,
                        help="Sila tlumienia swiatla 0..1 (0=surowo, 1=twarz mocno > swiatlo)")
    parser.add_argument("--eps", type=float, default=0.35, help="DBSCAN: promien sasiedztwa (kosinus)")
    parser.add_argument("--min-samples", type=int, default=2, help="DBSCAN: min. liczba probek")
    parser.add_argument("--threshold", type=float, default=0.6, help="Chinese Whispers: prog podobienstwa")
    args = parser.parse_args()

    emb_dir, faces_dir = Path(args.emb), Path(args.faces)
    names, X, _faces = load_embeddings(emb_dir, faces_dir)
    if len(names) == 0:
        print(f"Brak embeddingow w {emb_dir}")
        return

    if not args.no_normalize:
        X = l2_normalize(X)
    if args.light_strength > 0:
        X = regress_out(X, load_brightness(emb_dir, names), strength=args.light_strength)

    if args.method == "dbscan":
        labels = cluster_dbscan(X, eps=args.eps, min_samples=args.min_samples)
    else:
        labels = cluster_chinese_whispers(X, threshold=args.threshold)

    out = Path(args.out)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["name", "person"])
        for name, lab in zip(names, labels):
            person = "szum" if lab == -1 else f"osoba_{int(lab)}"
            writer.writerow([name, person])

    n_groups = len({l for l in labels if l != -1})
    n_noise = int(sum(1 for l in labels if l == -1))
    print(f"Metoda: {args.method} | twarzy: {len(names)} | grup: {n_groups} | szum: {n_noise}")
    print(f"Zapisano etykiety: {out}")


if __name__ == "__main__":
    main()
