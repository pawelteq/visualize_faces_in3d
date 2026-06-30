"""
Wspolne wczytywanie danych pipeline'u.

- load_embeddings         : embeddingi .npy + dopasowane podglady twarzy.
                            Twarze sa szukane REKURENCYJNIE - rowniez w podfolderach
                            osob (emb_images/<osoba>/...), bo tam trafiaja po
                            recznym etykietowaniu.
- load_brightness         : kowiata jasnosci z emb/meta.csv (do regresji swiatla).
- load_labels             : etykiety osob z pliku CSV (kolumny: name,person).
- load_labels_from_folders: etykiety z podfolderow emb_images/<osoba>/...
                            (foldery-nie-osoby, np. nierozpoznane/zwierzeta, sa pomijane).

Dopasowanie dziala dla nazw <nazwa>#<i>.npy oraz <nazwa>#<i>#<ts>.npy.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

# Foldery, ktore NIE oznaczaja osoby (smieci / zwierzeta / nierozpoznane).
DEFAULT_IGNORE = {
    "nierozpoznane", "nierozpoznani", "zwierzeta", "zwierzęta",
    "unknown", "inne", "other", "animals", "smieci", "śmieci",
}


def _index_faces(faces_dir: Path) -> dict:
    """Mapa stem -> sciezka do wycietej twarzy, takze z podfolderow osob."""
    index = {}
    if faces_dir.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img in faces_dir.rglob(ext):
                index.setdefault(img.stem, img)  # stem = <base>#twarz#<i>
    return index


def load_embeddings(emb_dir: Path, faces_dir: Path):
    emb_dir, faces_dir = Path(emb_dir), Path(faces_dir)
    face_index = _index_faces(faces_dir)

    files = sorted(emb_dir.glob("*.npy"))
    names, vectors, faces = [], [], []
    for f in files:
        parts = f.stem.split("#")
        key = f"{parts[0]}#twarz#{parts[1]}" if len(parts) >= 2 else f.stem
        img_path = face_index.get(key)
        if img_path is not None:
            img = np.array(Image.open(img_path).convert("RGB"))
        else:
            img = np.full((50, 50, 3), 200, dtype=np.uint8)  # placeholder gdy brak twarzy
        names.append(f.stem)
        vectors.append(np.load(f))
        faces.append(img)
    X = np.array(vectors) if vectors else np.empty((0, 0))
    return names, X, faces


def load_brightness(emb_dir: Path, names: list) -> np.ndarray:
    table = {}
    meta = Path(emb_dir) / "meta.csv"
    if meta.exists():
        with open(meta, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                table[row["name"]] = float(row["brightness"])
    return np.array([table.get(n, np.nan) for n in names])


def load_labels(path, names: list) -> list:
    table = {}
    if path and Path(path).exists():
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                table[row["name"]] = row["person"]
    return [table.get(n) for n in names]


def load_labels_from_folders(faces_dir, names: list, ignore_folders=None) -> list:
    """
    Czyta etykiety z ukladu folderow:
        emb_images/<osoba>/<base>#twarz#<i>.jpg
    Nazwa podfolderu = osoba. Foldery z ignore_folders (np. nierozpoznane,
    zwierzeta) oraz pliki luzem bez podfolderu -> bez etykiety (None).
    """
    ignore = DEFAULT_IGNORE if ignore_folders is None else {str(s).lower() for s in ignore_folders}
    mapping = {}
    fd = Path(faces_dir)
    if fd.exists():
        for sub in fd.iterdir():
            if not sub.is_dir() or sub.name.strip().lower() in ignore:
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for img in sub.rglob(ext):
                    stem = img.stem  # <base>#twarz#<i>
                    if "#twarz#" in stem:
                        base, idx = stem.split("#twarz#")
                        mapping[(base, idx)] = sub.name
    out = []
    for n in names:
        parts = n.split("#")
        out.append(mapping.get((parts[0], parts[1])) if len(parts) >= 2 else None)
    return out
