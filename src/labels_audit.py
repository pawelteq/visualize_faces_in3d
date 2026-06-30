"""
Wykrywanie prawdopodobnych pomylek w etykietach.

Dla kazdej oznaczonej twarzy porownuje podobienstwo do CENTROIDU wlasnej osoby
z podobienstwem do najblizszej INNEJ osoby. Jesli twarz jest blizej innej osoby
niz wlasnej - to kandydat na pomylke (albo bardzo trudne zdjecie).

Uzycie:
    python labels_audit.py                      # raport -> emb/suspected_mislabels.csv
    python labels_audit.py --min-face-size 64
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from dataset import load_embeddings, load_labels, load_labels_from_folders, DEFAULT_IGNORE
from postprocess import l2_normalize
from quality import size_mask, apply_mask


def _centroids(Xn, labels):
    persons = sorted({l for l in labels if l is not None})
    cents = {}
    for p in persons:
        idx = [i for i, l in enumerate(labels) if l == p]
        cents[p] = l2_normalize(Xn[idx].mean(axis=0, keepdims=True))[0]
    return cents


def suspected_mislabels(X, labels):
    """Lista krotek (idx, own, best_other, sim_own, sim_other) dla podejrzanych twarzy."""
    Xn = l2_normalize(X)
    cents = _centroids(Xn, labels)
    persons = list(cents)
    out = []
    for i, lab in enumerate(labels):
        if lab is None or lab not in cents:
            continue
        sim_own = float(Xn[i] @ cents[lab])
        others = [(p, float(Xn[i] @ cents[p])) for p in persons if p != lab]
        if not others:
            continue
        best_other, sim_other = max(others, key=lambda t: t[1])
        if sim_other > sim_own:
            out.append((i, lab, best_other, sim_own, sim_other))
    return out


def drop_suspected(labels, X):
    """Zwraca kopie etykiet z podejrzanymi ustawionymi na None (wykluczone z uczenia)."""
    susp = {i for i, *_ in suspected_mislabels(X, labels)}
    return [None if i in susp else l for i, l in enumerate(labels)], len(susp)


def find_project_root(start: Path) -> Path:
    for c in (start, start.parent):
        if (c / "emb").exists():
            return c
    return start


ROOT = find_project_root(Path(__file__).resolve().parent)


def main() -> None:
    ap = argparse.ArgumentParser(description="Wykrywanie pomylek w etykietach")
    ap.add_argument("--emb", default=str(ROOT / "emb"))
    ap.add_argument("--faces", default=str(ROOT / "emb_images"))
    ap.add_argument("--labels", default="", help="CSV; domyslnie podfoldery emb_images")
    ap.add_argument("--min-face-size", type=int, default=50)
    ap.add_argument("--ignore-folders", nargs="*", default=None,
                    help="Foldery-nie-osoby (domyslnie: nierozpoznane, zwierzeta, ...)")
    ap.add_argument("--out", default=str(ROOT / "emb" / "suspected_mislabels.csv"))
    args = ap.parse_args()

    emb_dir, faces_dir = Path(args.emb), Path(args.faces)
    ignore = DEFAULT_IGNORE if args.ignore_folders is None else {s.lower() for s in args.ignore_folders}

    names, X, faces = load_embeddings(emb_dir, faces_dir)
    if len(names) == 0:
        print("Brak embeddingow.")
        return
    names, X, faces = apply_mask(size_mask(faces, args.min_face_size), names, X, faces)

    labels = (load_labels(args.labels, names) if args.labels
              else load_labels_from_folders(faces_dir, names, ignore_folders=ignore))
    if len({l for l in labels if l is not None}) < 2:
        print("Potrzeba >=2 osob z etykietami.")
        return

    susp = suspected_mislabels(X, labels)
    out = Path(args.out)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "assigned", "closer_to", "sim_own", "sim_other"])
        for i, own, other, so, soo in sorted(susp, key=lambda t: t[3]):
            w.writerow([names[i], own, other, f"{so:.3f}", f"{soo:.3f}"])

    n_lab = len([l for l in labels if l is not None])
    print(f"Oznaczonych twarzy: {n_lab} | podejrzanych pomylek: {len(susp)}")
    print(f"Raport: {out}")
    for i, own, other, so, soo in sorted(susp, key=lambda t: t[3])[:10]:
        print(f"  {names[i]}: oznaczono '{own}', blizej '{other}' ({soo:.2f} > {so:.2f})")


if __name__ == "__main__":
    main()
