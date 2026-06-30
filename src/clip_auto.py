"""
Beznadzorowa kalibracja filtra CLIP - bez recznego ukladania przykladow.

Pomysl: CLIP daje kazdemu kadrowi przewage  margin = human - reject.
Prawdziwe twarze maja ja wysoka, smieci (psy, rendery, grafiki) niska/ujemna.
Te dwa skupiska rozdziela sie AUTOMATYCZNIE (Otsu / mieszanka gaussowska),
wiec prog dobiera sie sam, bez etykiet.

Skrypt liczy CLIP raz, wyznacza ciecie, raportuje statystyki i buduje
clip_tuning.html juz USTAWIONY na znalezionej granicy (mozesz ja jeszcze
recznie podkrecic suwakiem).

Uzycie:
    python clip_auto.py --source images --device gpu
    python clip_auto.py --source crops --dir emb_images --device gpu
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from clip_export import (make_thumb, build_html, candidates_from_crops,
                         candidates_from_images)


def find_project_root(start: Path) -> Path:
    for c in (start, start.parent):
        if (c / "emb_images").exists() or (c / "images").exists():
            return c
    return start


ROOT = find_project_root(Path(__file__).resolve().parent)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Klasyczne Otsu na 1D: prog maksymalizujacy wariancje miedzy dwoma grupami."""
    v = np.asarray(values, dtype=float)
    hist, edges = np.histogram(v, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return float(np.median(v))
    w = np.cumsum(hist).astype(float)             # liczba probek <= prog
    wb = w / total                                # waga "tla" (ponizej)
    wf = 1.0 - wb                                 # waga "obiektu" (powyzej)
    csum = np.cumsum(hist * centers)              # suma wartosci <= prog
    total_sum = csum[-1]
    mb = np.divide(csum, w, out=np.zeros_like(csum), where=w > 0)
    mf = np.divide(total_sum - csum, (total - w),
                   out=np.zeros_like(csum), where=(total - w) > 0)
    between = wb * wf * (mb - mf) ** 2
    return float(centers[int(np.argmax(between))])


def gmm_threshold(values: np.ndarray):
    """Mieszanka 2 gaussow -> granica miedzy skupiskami. Zwraca (cut, info) lub None."""
    try:
        from sklearn.mixture import GaussianMixture
    except Exception:  # noqa: BLE001
        return None
    v = np.asarray(values, dtype=float).reshape(-1, 1)
    g = GaussianMixture(n_components=2, random_state=42).fit(v)
    means = sorted(float(m) for m in g.means_.ravel())
    lo, hi = means
    if hi - lo < 1e-6:
        return None
    ts = np.linspace(lo, hi, 600).reshape(-1, 1)
    lab = g.predict(ts)
    flips = np.where(np.diff(lab) != 0)[0]
    cut = float(ts[flips[0], 0]) if len(flips) else float((lo + hi) / 2)
    return cut, {"mean_low": lo, "mean_high": hi}


def main() -> None:
    ap = argparse.ArgumentParser(description="Beznadzorowa kalibracja filtra CLIP")
    ap.add_argument("--source", choices=["images", "crops"], default="images")
    ap.add_argument("--images", default=str(ROOT / "images"))
    ap.add_argument("--dir", default=str(ROOT / "emb_images"))
    ap.add_argument("--out", default=str(ROOT / "clip_tuning.html"))
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--max-images", type=int, default=600)
    ap.add_argument("--thumb", type=int, default=84)
    ap.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--min-score", type=float, default=0.6)
    ap.add_argument("--min-detect-size", type=int, default=40)
    ap.add_argument("--clip-model", default="ViT-B-32")
    ap.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--method", choices=["gmm", "otsu"], default="gmm")
    args = ap.parse_args()

    if args.source == "images":
        print(f"Detekcja kandydatur z {args.images} ...")
        candidates = candidates_from_images(Path(args.images), args)
    else:
        candidates = candidates_from_crops(Path(args.dir), args.limit)
    if len(candidates) < 5:
        print(f"Za malo kadrow ({len(candidates)}).")
        return
    print(f"Kadrow: {len(candidates)}. Licze CLIP...")

    from face_filter import ClipFaceFilter
    clip_device = "cpu" if args.device == "cpu" else "auto"
    flt = ClipFaceFilter(device=clip_device, model_name=args.clip_model,
                         pretrained=args.clip_pretrained)

    items, margins = [], []
    for n, (name, crop) in enumerate(candidates, 1):
        _ok, h, r = flt.scores(crop)
        items.append({"name": name, "img": make_thumb(crop, args.thumb),
                      "h": round(float(h), 4), "r": round(float(r), 4)})
        margins.append(h - r)
        if n % 100 == 0:
            print(f"  CLIP {n}/{len(candidates)}")
    margins = np.array(margins)

    otsu = otsu_threshold(margins)
    gmm = gmm_threshold(margins) if args.method == "gmm" else None
    if gmm is not None:
        cut, info = gmm
        method = f"GMM (skupiska ~{info['mean_low']:.3f} i {info['mean_high']:.3f})"
    else:
        cut, method = otsu, "Otsu"

    keep = margins >= cut
    print("\n=== AUTOMATYCZNE CIECIE (bez etykiet) ===")
    print(f"  metoda: {method}")
    print(f"  ciecie przewagi (human-reject) = {cut:.3f}")
    print(f"  Otsu (porownawczo) = {otsu:.3f}")
    print(f"  zostaje: {int(keep.sum())} | wypada: {int((~keep).sum())} "
          f"({100*(~keep).mean():.1f}%)")
    sep = margins[keep].mean() - margins[~keep].mean() if keep.any() and (~keep).any() else 0.0
    print(f"  rozdzielenie skupisk: {sep:.3f} "
          f"({'wyrazne' if sep > 0.06 else 'slabe - rozwaz mocniejszy model --clip-model ViT-L-14'})")
    print(f"\nUzyj w detekcji:  python src/detect_faces.py --device gpu --rebuild "
          f"--min-score 0.7 --clip-filter --clip-threshold 0 --clip-margin {cut:.3f}")

    html = build_html(items, 0.0, round(float(cut), 3))
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"\nStrona z auto-ustawieniem: {args.out} (otworz i ewentualnie dostroj suwakiem)")


if __name__ == "__main__":
    main()
