"""
Wizualizacja 3D embeddingow twarzy (PyVista).

Wczytuje embeddingi z emb/, redukuje wymiarowosc do 3D (PCA lub t-SNE)
i renderuje kazda twarz jako teksturowany kafelek w przestrzeni 3D.
Twarze blisko siebie = podobne (potencjalnie ta sama osoba).

Dopasowanie obrazu do embeddingu dziala dla nazw:
    emb/<nazwa>#<i>.npy            (nowy, deterministyczny format)
    emb/<nazwa>#<i>#<timestamp>.npy (stary format - tez obslugiwany)

Przyklady:
    python visualize_3d.py
    python visualize_3d.py --method TSNE
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def find_project_root(start: Path) -> Path:
    for candidate in (start, start.parent):
        if (candidate / "emb").exists():
            return candidate
    return start


ROOT = find_project_root(Path(__file__).resolve().parent)


def load_embeddings(emb_dir: Path, faces_dir: Path):
    files = sorted(emb_dir.glob("*.npy"))
    if not files:
        print(f"Brak embeddingow w folderze: {emb_dir}")
        return None

    vectors, labels, faces = [], [], []
    for f in files:
        parts = f.stem.split("#")  # [nazwa, indeks, (timestamp)]
        face_path = faces_dir / f"{parts[0]}#twarz#{parts[1]}.jpg"
        if face_path.exists():
            img = np.array(Image.open(face_path).convert("RGB"))
        else:
            img = np.full((50, 50, 3), 200, dtype=np.uint8)
            print(f"Brak wycietej twarzy: {face_path.name} (uzywam szarego kafelka)")

        vectors.append(np.load(f))
        labels.append(f.stem)
        faces.append(img)

    return np.array(vectors), labels, faces


def reduce_to_3d(X: np.ndarray, method: str):
    method = method.upper()
    if method == "PCA":
        coords = PCA(n_components=3).fit_transform(X) * 5.0
        return coords, "Embeddingi twarzy - PCA"
    if method == "TSNE":
        perplexity = max(1, min(30, len(X) - 1))
        coords = TSNE(n_components=3, random_state=42, perplexity=perplexity).fit_transform(X)
        return coords, "Embeddingi twarzy - t-SNE"
    raise ValueError(f"Nieznana metoda redukcji: {method} (dostepne: PCA, TSNE)")


def visualize(coords: np.ndarray, faces: list, title: str, tile: float = 0.12) -> None:
    plotter = pv.Plotter()
    for center, img in zip(coords, faces):
        plane = pv.Plane(center=center, direction=(0, 0, 1), i_size=tile, j_size=tile)
        plotter.add_mesh(plane, texture=pv.Texture(img))
    plotter.add_axes()
    plotter.add_text(title, font_size=12)
    plotter.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Wizualizacja 3D embeddingow twarzy")
    parser.add_argument("--method", choices=["PCA", "TSNE"], default="PCA")
    parser.add_argument("--emb", default=str(ROOT / "emb"))
    parser.add_argument("--faces", default=str(ROOT / "emb_images"))
    args = parser.parse_args()

    data = load_embeddings(Path(args.emb), Path(args.faces))
    if data is None:
        return

    X, _labels, faces = data
    if len(X) < 3:
        print(f"Za malo embeddingow do redukcji 3D (jest {len(X)}, trzeba >= 3).")
        return

    print(f"Wczytano {len(X)} embeddingow (wymiar {X.shape[1]}). Redukcja: {args.method}...")
    coords, title = reduce_to_3d(X, args.method)
    visualize(coords, faces, title)


if __name__ == "__main__":
    main()
