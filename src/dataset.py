"""
Wspolne wczytywanie danych pipeline'u.

- load_embeddings         : embeddingi .npy + dopasowane podglady twarzy.
                            Twarze sa szukane REKURENCYJNIE - rowniez w podfolderach
                            osob (emb_images/<osoba>/...), bo tam trafiaja po
                            recznym etykietowaniu.
- face_folders            : dla kazdego embeddingu nazwa podfolderu, w ktorym lezy
                            jego twarz (sluzy do wykluczenia zwierzat/nierozpoznanych
                            z wizualizacji).
- load_brightness         : kowiata jasnosci z emb/meta.csv (do regresji swiatla).
- load_labels             : etykiety osob z pliku CSV (kolumny: name,person).
- load_labels_from_folders: etykiety z podfolderow emb_images/<osoba>/...

Dopasowanie dziala dla nazw <nazwa>#<i>.npy oraz <nazwa>#<i>#<ts>.npy.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

# Foldery, ktore NIE oznaczaja osoby (smieci / zwierzeta / nierozpoznane).
DEFAULT_IGNORE = {
    "nierozpoznane", "zwierzeta", "zwierzęta",
    "unknown", "inne", "other", "animals", "smieci", "śmieci",
}


def _face_key(name: str) -> str:
    parts = name.split("#")
    return f"{parts[0]}#twarz#{parts[1]}" if len(parts) >= 2 else name


def _index_faces(faces_dir: Path):
    """Mapy stem -> (sciezka, nazwa_podfolderu) dla wszystkich crop'ow (tez w podfolderach)."""
    path_index, folder_index = {}, {}
    if faces_dir.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img in faces_dir.rglob(ext):
                if img.stem in path_index:
                    continue
                rel = img.relative_to(faces_dir)
                path_index[img.stem] = img
                folder_index[img.stem] = rel.parts[0] if len(rel.parts) > 1 else None
    return path_index, folder_index


def load_embeddings(emb_dir: Path, faces_dir: Path):
    emb_dir, faces_dir = Path(emb_dir), Path(faces_dir)
    path_index, _ = _index_faces(faces_dir)

    files = sorted(emb_dir.glob("*.npy"))
    names, vectors, faces = [], [], []
    for f in files:
        img_path = path_index.get(_face_key(f.stem))
        if img_path is not None:
            img = np.array(Image.open(img_path).convert("RGB"))
        else:
            img = np.full((50, 50, 3), 200, dtype=np.uint8)  # placeholder gdy brak twarzy
        names.append(f.stem)
        vectors.append(np.load(f))
        faces.append(img)
    X = np.array(vectors) if vectors else np.empty((0, 0))
    return names, X, faces


def face_folders(faces_dir, names: list) -> list:
    """Nazwa podfolderu, w ktorym lezy twarz danego embeddingu (None = luzem/brak)."""
    _, folder_index = _index_faces(Path(faces_dir))
    return [folder_index.get(_face_key(n)) for n in names]


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
    Etykiety z ukladu folderow: emb_images/<osoba>/<base>#twarz#<i>.jpg
    Foldery z ignore_folders (np. nierozpoznane, zwierzeta) i pliki luzem -> None.
    """
    ignore = DEFAULT_IGNORE if ignore_folders is None else {str(s).lower() for s in ignore_folders}
    folders = face_folders(faces_dir, names)
    out = []
    for f in folders:
        out.append(None if (f is None or f.strip().lower() in ignore) else f)
    return out
