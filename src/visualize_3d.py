"""
Wizualizacja 3D embeddingow twarzy (PyVista).

Twarz wazy mocniej niz swiatlo (L2-normalizacja + tlumienie swiatla domyslnie),
a uklad jest tak dobrany, by PODOBNE twarze byly blisko, a GRUPY mocno
odseparowane. Z etykietami mozesz NAUCZYC metryki, ktora sciaga ta sama osobe
razem i odpycha rozne osoby.

Odporne na realny zbior: pomija foldery-nie-osoby (nierozpoznane, zwierzeta...),
odrzuca male kadry (--min-face-size), moze pominac pomylki (--drop-suspected).

  Uczenie metryki (wymaga etykiet):
    --metric          none | lda | nca | lmnn
    --labels-from-folders   etykiety z podfolderow emb_images/<osoba>/
    --drop-suspected        pomin prawdopodobne pomylki w etykietach
  Geometria / rozrzut:
    --reduce          PCA | TSNE | UMAP | LDA   (gdy --metric none)
    --spread / --explode / --perplexity
  Jakosc / filtry:
    --min-face-size   odrzuc kadry mniejsze niz N px (domyslnie 50)
    --ignore-folders  nazwy folderow-nie-osob
  Anty-swiatlo: --light-strength 0..1 / --no-normalize
  Kolor: --color-by none | cluster | label    --eps (mniejsze=wiecej grup)

Przyklady:
    python visualize_3d.py --reduce UMAP --explode 2.5
    python visualize_3d.py --metric lmnn --labels-from-folders --drop-suspected --explode 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

from dataset import (load_embeddings, load_brightness, load_labels,
                     load_labels_from_folders, DEFAULT_IGNORE)
from postprocess import l2_normalize, regress_out, lda_reduce
from metric_learning import learn_metric
from quality import size_mask, apply_mask
from labels_audit import drop_suspected


def find_project_root(start: Path) -> Path:
    for candidate in (start, start.parent):
        if (candidate / "emb").exists():
            return candidate
    return start


ROOT = find_project_root(Path(__file__).resolve().parent)


def reduce_to_3d(X, method, labels, perplexity, umap_neighbors, umap_min_dist):
    method = method.upper()
    if method == "PCA":
        from sklearn.decomposition import PCA
        return PCA(n_components=3).fit_transform(X), "PCA"
    if method == "TSNE":
        from sklearn.manifold import TSNE
        perp = perplexity if perplexity and perplexity > 0 else max(1, min(30, len(X) - 1))
        perp = min(perp, len(X) - 1)
        return TSNE(n_components=3, random_state=42, perplexity=perp,
                    init="pca").fit_transform(X), f"t-SNE (perp={perp:g})"
    if method == "UMAP":
        try:
            import umap
        except ImportError as exc:
            raise SystemExit("UMAP wymaga: pip install umap-learn") from exc
        reducer = umap.UMAP(n_components=3, metric="cosine",
                            n_neighbors=max(2, min(umap_neighbors, len(X) - 1)),
                            min_dist=umap_min_dist, random_state=42)
        return reducer.fit_transform(X), "UMAP"
    if method == "LDA":
        if labels is None or any(l is None for l in labels):
            raise SystemExit("LDA wymaga pelnych etykiet (--labels / --labels-from-folders).")
        return lda_reduce(X, labels, n_components=3), "LDA"
    raise ValueError(f"Nieznana metoda: {method}")


def standardize_and_spread(coords, spread):
    c = np.asarray(coords, dtype=float)
    c = c - c.mean(axis=0)
    std = c.std()
    if std > 0:
        c = c / std
    return c * spread


def explode_clusters(coords, labels, factor):
    """Rozpycha grupy: kazdy klaster odsuwany od srodka. Ksztalt w srodku grupy zostaje."""
    if factor <= 1 or labels is None:
        return coords
    out = np.array(coords, dtype=float)
    global_center = out.mean(axis=0)
    for lab in set(labels):
        if lab in (-1, None):
            continue
        idx = np.array([l == lab for l in labels])
        centroid = out[idx].mean(axis=0)
        out[idx] += (centroid - global_center) * (factor - 1.0)
    return out


def groups_to_ids(groups):
    uniq, ids = {}, []
    for g in groups:
        key = "?" if g is None else g
        if key not in uniq:
            uniq[key] = len(uniq)
        ids.append(uniq[key])
    return np.array(ids, dtype=float)


def count_groups(labels):
    if labels is None:
        return 0
    return len({l for l in labels if l not in (-1, None)})


def visualize(coords, faces, color_ids, title, tile):
    plotter = pv.Plotter()
    for center, img in zip(coords, faces):
        plane = pv.Plane(center=center, direction=(0, 0, 1), i_size=tile, j_size=tile)
        plotter.add_mesh(plane, texture=pv.Texture(img))
    if color_ids is not None:
        cloud = pv.PolyData(np.asarray(coords, dtype=float))
        cloud["osoba"] = color_ids
        plotter.add_mesh(cloud, scalars="osoba", render_points_as_spheres=True,
                         point_size=18, cmap="tab20", show_scalar_bar=False)
    plotter.add_axes()
    plotter.add_text(title, font_size=12)
    plotter.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Wizualizacja 3D embeddingow twarzy")
    parser.add_argument("--metric", choices=["none", "lda", "nca", "lmnn"], default="none",
                        help="Uczenie metryki z etykiet (sciaga te sama osobe, odpycha rozne)")
    parser.add_argument("--labels-from-folders", action="store_true",
                        help="Etykiety z podfolderow emb_images/<osoba>/")
    parser.add_argument("--drop-suspected", action="store_true",
                        help="Pomin prawdopodobne pomylki w etykietach")
    parser.add_argument("--reduce", choices=["PCA", "TSNE", "UMAP", "LDA"], default="UMAP")
    parser.add_argument("--color-by", choices=["none", "cluster", "label"], default="cluster")
    parser.add_argument("--spread", type=float, default=10.0, help="Globalna skala (wieksze odleglosci)")
    parser.add_argument("--explode", type=float, default=2.0, help="Rozpychanie grup (>1)")
    parser.add_argument("--perplexity", type=float, default=0.0, help="t-SNE: nizsza -> wiecej grup")
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--no-normalize", action="store_true", help="Wylacz L2-normalizacje")
    parser.add_argument("--light-strength", type=float, default=1.0,
                        help="Sila tlumienia swiatla 0..1 (0=surowo, 1=twarz mocno > swiatlo)")
    parser.add_argument("--min-face-size", type=int, default=50,
                        help="Odrzuc kadry mniejsze niz N px (0 = bez filtra)")
    parser.add_argument("--ignore-folders", nargs="*", default=None,
                        help="Foldery-nie-osoby (domyslnie: nierozpoznane, zwierzeta, ...)")
    parser.add_argument("--labels", default=str(ROOT / "emb" / "labels_auto.csv"))
    parser.add_argument("--emb", default=str(ROOT / "emb"))
    parser.add_argument("--faces", default=str(ROOT / "emb_images"))
    parser.add_argument("--eps", type=float, default=0.30, help="DBSCAN: mniejsze -> wiecej grup")
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--tile-scale", type=float, default=1.0, help="Rozmiar miniatur twarzy")
    args = parser.parse_args()

    emb_dir, faces_dir = Path(args.emb), Path(args.faces)
    ignore = DEFAULT_IGNORE if args.ignore_folders is None else {s.lower() for s in args.ignore_folders}

    names, X, faces = load_embeddings(emb_dir, faces_dir)
    if len(names) == 0:
        print(f"Brak embeddingow w {emb_dir}")
        return
    before = len(names)
    names, X, faces = apply_mask(size_mask(faces, args.min_face_size), names, X, faces)
    if len(X) < 3:
        print(f"Za malo embeddingow po filtrze rozmiaru (jest {len(X)}, trzeba >= 3).")
        return

    if not args.no_normalize:
        X = l2_normalize(X)
    if args.light_strength > 0:
        X = regress_out(X, load_brightness(emb_dir, names), strength=args.light_strength)

    # Etykiety (potrzebne do metryki, LDA, lub koloru wg osoby)
    need_labels = args.metric != "none" or args.reduce == "LDA" or args.color_by == "label"
    labels = None
    if need_labels:
        labels = (load_labels_from_folders(faces_dir, names, ignore_folders=ignore)
                  if args.labels_from_folders else load_labels(args.labels, names))
        if args.drop_suspected and labels is not None:
            labels, n_susp = drop_suspected(labels, X)
            print(f"Pominieto {n_susp} podejrzanych etykiet.")

    # Wspolrzedne 3D
    if args.metric != "none":
        if labels is None or all(l is None for l in labels):
            raise SystemExit("--metric wymaga etykiet (--labels-from-folders lub --labels).")
        coords = learn_metric(X, labels, method=args.metric, n_components=3)
        label_method = f"metryka:{args.metric}"
    else:
        coords, label_method = reduce_to_3d(X, args.reduce, labels,
                                            args.perplexity, args.umap_neighbors, args.umap_min_dist)

    # Grupy do koloru i rozpychania
    if args.color_by == "label" or args.metric != "none":
        group_labels = labels
    elif args.color_by == "cluster":
        from cluster import cluster_dbscan
        group_labels = cluster_dbscan(X, eps=args.eps, min_samples=args.min_samples).tolist()
    else:
        group_labels = None
    color_ids = groups_to_ids(group_labels) if group_labels is not None else None

    coords = standardize_and_spread(coords, args.spread)
    coords = explode_clusters(coords, group_labels, args.explode)

    extent = float(np.ptp(coords, axis=0).max()) or 1.0
    tile = extent * 0.03 * args.tile_scale

    n_groups = count_groups(group_labels)
    print(f"Twarzy: {before} -> po filtrze {len(names)} | uklad: {label_method}"
          f"{'' if args.no_normalize else ' + L2'}"
          f"{f' + swiatlo x{args.light_strength:g}' if args.light_strength > 0 else ''}"
          f" | grup: {n_groups} | spread={args.spread:g} explode={args.explode:g}")
    title = f"Embeddingi twarzy - {label_method} ({n_groups} grup)"
    visualize(coords, faces, color_ids, title, tile)


if __name__ == "__main__":
    main()
