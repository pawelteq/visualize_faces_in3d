"""
Wizualizacja 3D embeddingow twarzy (PyVista) - interaktywna.

Twarz wazy mocniej niz swiatlo (L2-normalizacja + tlumienie swiatla domyslnie).
Te same osoby sa grupowane, a w obrebie grupy zdjecia ukladane sa na SIATCE,
zeby kazde bylo widoczne (nie nakladaly sie na siebie).

INTERAKCJA:
    - kliknij dowolne zdjecie -> kamera przybliza do calej grupy tej osoby,
      u gory pojawia sie nazwa osoby i liczba zdjec,
    - klawisz R -> reset widoku.

Odporne na realny zbior: pomija zwierzeta/nierozpoznane, male kadry,
opcjonalnie pomylki w etykietach.

Przyklady:
    python visualize_3d.py --metric lmnn --labels-from-folders --only-labeled
    python visualize_3d.py --metric lmnn --labels-from-folders --cell 0.8 --explode 3
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pyvista as pv

from dataset import (load_embeddings, face_folders, load_brightness,
                     load_labels, load_labels_from_folders, DEFAULT_IGNORE)
from postprocess import l2_normalize, regress_out, lda_reduce
from metric_learning import learn_metric
from quality import size_mask, folder_keep_mask, apply_mask
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


def layout_groups(coords, labels, cell, explode):
    """Kazda grupe rozsuwa od srodka (explode) i uklada jej zdjecia na SIATCE
    (cell = odstep), zeby sie nie nakladaly. Punkty bez grupy zostaja na miejscu."""
    coords = np.asarray(coords, dtype=float).copy()
    if labels is None:
        return coords
    out = coords.copy()
    global_center = coords.mean(axis=0)
    for lab in sorted({l for l in labels if l not in (-1, None)}, key=str):
        idx = [i for i, l in enumerate(labels) if l == lab]
        centroid = coords[idx].mean(axis=0)
        exploded = centroid + (centroid - global_center) * (explode - 1.0)
        n = len(idx)
        cols = max(1, math.ceil(math.sqrt(n)))
        rows = math.ceil(n / cols)
        for k, i in enumerate(idx):
            r, c = divmod(k, cols)
            offset = np.array([(c - (cols - 1) / 2) * cell,
                               ((rows - 1) / 2 - r) * cell, 0.0])
            out[i] = exploded + offset
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


def _enable_group_zoom(plotter, coords, labels, cell):
    """Klik w zdjecie -> kamera przybliza do calej grupy tej osoby."""
    coords = np.asarray(coords, dtype=float)

    def on_pick(point, *_):
        if point is None:
            return
        i = int(np.argmin(np.linalg.norm(coords - np.asarray(point), axis=1)))
        lab = labels[i] if labels is not None else None
        members = [j for j, l in enumerate(labels) if l == lab] if lab is not None else [i]
        pts = coords[members]
        center = pts.mean(axis=0)
        extent = float(np.linalg.norm(np.ptp(pts, axis=0))) or cell

        cam = plotter.camera
        d = np.array(cam.position) - np.array(cam.focal_point)
        nrm = np.linalg.norm(d)
        d = d / nrm if nrm > 0 else np.array([0.0, 0.0, 1.0])
        dist = extent * 1.4 + cell * 6.0
        cam.focal_point = center.tolist()
        cam.position = (center + d * dist).tolist()
        plotter.add_text(f"Osoba: {lab}  ({len(members)} zdjec)   [R = reset]",
                         name="pickinfo", font_size=11)
        plotter.render()

    try:
        plotter.enable_point_picking(callback=on_pick, left_clicking=True,
                                     show_point=False, show_message=False)
    except TypeError:  # starsze API PyVista
        plotter.enable_point_picking(callback=on_pick)


def visualize(coords, faces, color_ids, group_labels, title, tile, cell, interactive):
    plotter = pv.Plotter()
    for center, img in zip(coords, faces):
        plane = pv.Plane(center=center, direction=(0, 0, 1), i_size=tile, j_size=tile)
        plotter.add_mesh(plane, texture=pv.Texture(img))
    if color_ids is not None:
        cloud = pv.PolyData(np.asarray(coords, dtype=float))
        cloud["osoba"] = color_ids
        plotter.add_mesh(cloud, scalars="osoba", render_points_as_spheres=True,
                         point_size=10, cmap="tab20", show_scalar_bar=False)
    plotter.add_axes()
    plotter.add_text(title, font_size=12)
    if interactive and group_labels is not None:
        plotter.add_text("Kliknij osobe, aby przyblizyc  |  R = reset",
                         position="lower_left", font_size=9, name="hint")
        _enable_group_zoom(plotter, coords, group_labels, cell)
    plotter.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interaktywna wizualizacja 3D embeddingow twarzy")
    parser.add_argument("--metric", choices=["none", "lda", "nca", "lmnn"], default="none")
    parser.add_argument("--labels-from-folders", action="store_true")
    parser.add_argument("--drop-suspected", action="store_true")
    parser.add_argument("--reduce", choices=["PCA", "TSNE", "UMAP", "LDA"], default="UMAP")
    parser.add_argument("--color-by", choices=["none", "cluster", "label"], default="cluster")
    parser.add_argument("--spread", type=float, default=12.0, help="Globalna skala (odleglosci miedzy grupami)")
    parser.add_argument("--explode", type=float, default=2.5, help="Rozpychanie grup (>1)")
    parser.add_argument("--cell", type=float, default=0.6, help="Odstep zdjec w obrebie grupy (siatka)")
    parser.add_argument("--perplexity", type=float, default=0.0)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--light-strength", type=float, default=1.0)
    parser.add_argument("--min-face-size", type=int, default=50)
    parser.add_argument("--ignore-folders", nargs="*", default=None)
    parser.add_argument("--only-labeled", action="store_true",
                        help="Pokaz tylko twarze ze sklasyfikowanych folderow osob")
    parser.add_argument("--show-ignored", action="store_true")
    parser.add_argument("--no-interactive", action="store_true", help="Wylacz klikanie/przyblizanie")
    parser.add_argument("--labels", default=str(ROOT / "emb" / "labels_auto.csv"))
    parser.add_argument("--emb", default=str(ROOT / "emb"))
    parser.add_argument("--faces", default=str(ROOT / "emb_images"))
    parser.add_argument("--eps", type=float, default=0.30)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--tile-scale", type=float, default=1.0)
    args = parser.parse_args()

    emb_dir, faces_dir = Path(args.emb), Path(args.faces)
    ignore = DEFAULT_IGNORE if args.ignore_folders is None else {s.lower() for s in args.ignore_folders}

    names, X, faces = load_embeddings(emb_dir, faces_dir)
    if len(names) == 0:
        print(f"Brak embeddingow w {emb_dir}")
        return
    before = len(names)

    names, X, faces = apply_mask(size_mask(faces, args.min_face_size), names, X, faces)
    folders = face_folders(faces_dir, names)
    keep_ignore = set() if args.show_ignored else ignore
    names, X, faces = apply_mask(
        folder_keep_mask(folders, keep_ignore, only_labeled=args.only_labeled),
        names, X, faces)

    if len(X) < 3:
        print(f"Za malo twarzy po filtrach (jest {len(X)}, trzeba >= 3).")
        return

    if not args.no_normalize:
        X = l2_normalize(X)
    if args.light_strength > 0:
        X = regress_out(X, load_brightness(emb_dir, names), strength=args.light_strength)

    need_labels = args.metric != "none" or args.reduce == "LDA" or args.color_by == "label"
    labels = None
    if need_labels:
        labels = (load_labels_from_folders(faces_dir, names, ignore_folders=ignore)
                  if args.labels_from_folders else load_labels(args.labels, names))
        if args.drop_suspected and labels is not None:
            labels, n_susp = drop_suspected(labels, X)
            print(f"Pominieto {n_susp} podejrzanych etykiet.")

    if args.metric != "none":
        if labels is None or all(l is None for l in labels):
            raise SystemExit("--metric wymaga etykiet (--labels-from-folders lub --labels).")
        coords = learn_metric(X, labels, method=args.metric, n_components=3)
        label_method = f"metryka:{args.metric}"
    else:
        coords, label_method = reduce_to_3d(X, args.reduce, labels,
                                            args.perplexity, args.umap_neighbors, args.umap_min_dist)

    if args.color_by == "label" or args.metric != "none":
        group_labels = labels
    elif args.color_by == "cluster":
        from cluster import cluster_dbscan
        group_labels = cluster_dbscan(X, eps=args.eps, min_samples=args.min_samples).tolist()
    else:
        group_labels = None
    color_ids = groups_to_ids(group_labels) if group_labels is not None else None

    coords = standardize_and_spread(coords, args.spread)
    coords = layout_groups(coords, group_labels, args.cell, args.explode)

    if group_labels is not None:
        tile = args.cell * 0.85 * args.tile_scale
    else:
        tile = (float(np.ptp(coords, axis=0).max()) or 1.0) * 0.03 * args.tile_scale

    n_groups = count_groups(group_labels)
    print(f"Twarzy: {before} -> po filtrach {len(names)} | uklad: {label_method}"
          f" | grup: {n_groups} | spread={args.spread:g} explode={args.explode:g} cell={args.cell:g}")
    if not args.no_interactive and group_labels is not None:
        print("Kliknij osobe w oknie, aby przyblizyc. Klawisz R resetuje widok.")
    title = f"Embeddingi twarzy - {label_method} ({n_groups} grup)"
    visualize(coords, faces, color_ids, group_labels, title, tile, args.cell,
              interactive=not args.no_interactive)


if __name__ == "__main__":
    main()
