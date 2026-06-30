"""
Wizualizacja 3D embeddingow twarzy (PyVista) z opcjami "anty-swiatlo".

Domyslnie twarz wazy mocniej niz swiatlo (L2-normalizacja + tlumienie swiatla
wlaczone). Sterujesz tym flagami:
    --light-strength  0..1  ile swiatla stlumic (0=surowo, 1=twarz mocno > swiatlo)
    --no-normalize          wylacz L2-normalizacje
    --reduce          PCA | TSNE | LDA   (LDA wymaga --labels)
    --color-by        none | cluster | label   (kolor punktu = osoba/grupa)

Przyklady:
    python visualize_3d.py                          # domyslnie: twarz > swiatlo, kolor=grupa
    python visualize_3d.py --light-strength 0       # surowe embeddingi (porownanie)
    python visualize_3d.py --reduce TSNE
    python visualize_3d.py --reduce LDA --labels emb/labels_auto.csv --color-by label
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

from dataset import load_embeddings, load_brightness, load_labels
from postprocess import l2_normalize, regress_out, lda_reduce


def find_project_root(start: Path) -> Path:
    for candidate in (start, start.parent):
        if (candidate / "emb").exists():
            return candidate
    return start


ROOT = find_project_root(Path(__file__).resolve().parent)


def reduce_to_3d(X, method, labels=None):
    method = method.upper()
    if method == "PCA":
        from sklearn.decomposition import PCA
        return PCA(n_components=3).fit_transform(X) * 5.0, "PCA"
    if method == "TSNE":
        from sklearn.manifold import TSNE
        perplexity = max(1, min(30, len(X) - 1))
        return TSNE(n_components=3, random_state=42, perplexity=perplexity).fit_transform(X), "t-SNE"
    if method == "LDA":
        if labels is None or any(l is None for l in labels):
            raise SystemExit("LDA wymaga pelnych etykiet (--labels emb/labels_auto.csv).")
        return lda_reduce(X, labels, n_components=3), "LDA"
    raise ValueError(f"Nieznana metoda: {method}")


def groups_to_ids(groups):
    """Mapuje etykiety (str/int/None) na liczby do kolorowania."""
    uniq = {}
    ids = []
    for g in groups:
        key = "?" if g is None else g
        if key not in uniq:
            uniq[key] = len(uniq)
        ids.append(uniq[key])
    return np.array(ids, dtype=float)


def visualize(coords, faces, color_ids, title, tile=0.12):
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
    parser.add_argument("--reduce", choices=["PCA", "TSNE", "LDA"], default="PCA")
    parser.add_argument("--color-by", choices=["none", "cluster", "label"], default="cluster")
    parser.add_argument("--no-normalize", action="store_true", help="Wylacz L2-normalizacje")
    parser.add_argument("--light-strength", type=float, default=1.0,
                        help="Sila tlumienia swiatla 0..1 (0=surowo, 1=twarz mocno > swiatlo)")
    parser.add_argument("--labels", default=str(ROOT / "emb" / "labels_auto.csv"))
    parser.add_argument("--emb", default=str(ROOT / "emb"))
    parser.add_argument("--faces", default=str(ROOT / "emb_images"))
    parser.add_argument("--eps", type=float, default=0.35)
    parser.add_argument("--min-samples", type=int, default=2)
    args = parser.parse_args()

    emb_dir, faces_dir = Path(args.emb), Path(args.faces)
    names, X, faces = load_embeddings(emb_dir, faces_dir)
    if len(names) == 0:
        print(f"Brak embeddingow w {emb_dir}")
        return
    if len(X) < 3:
        print(f"Za malo embeddingow do redukcji 3D (jest {len(X)}, trzeba >= 3).")
        return

    if not args.no_normalize:
        X = l2_normalize(X)
    if args.light_strength > 0:
        X = regress_out(X, load_brightness(emb_dir, names), strength=args.light_strength)

    labels = load_labels(args.labels, names) if args.reduce == "LDA" or args.color_by == "label" else None

    color_ids = None
    if args.color_by == "label":
        color_ids = groups_to_ids(labels)
    elif args.color_by == "cluster":
        from cluster import cluster_dbscan
        color_ids = groups_to_ids(cluster_dbscan(X, eps=args.eps, min_samples=args.min_samples).tolist())

    print(f"Wczytano {len(X)} embeddingow (wymiar {X.shape[1]}). Redukcja: {args.reduce}"
          f"{'' if args.no_normalize else ' + L2'}"
          f"{f' + tlumienie swiatla x{args.light_strength:g}' if args.light_strength > 0 else ''}")
    coords, label_method = reduce_to_3d(X, args.reduce, labels)
    title = f"Embeddingi twarzy - {label_method}"
    visualize(coords, faces, color_ids, title)


if __name__ == "__main__":
    main()
