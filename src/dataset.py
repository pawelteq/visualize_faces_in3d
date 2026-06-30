"""
Wspolne wczytywanie danych pipeline'u.

- load_embeddings : embeddingi .npy + dopasowane podglady twarzy.
- load_brightness : kowiata jasnosci z emb/meta.csv (do regresji swiatla).
- load_labels     : etykiety osob z pliku CSV (kolumny: name,person).

Dopasowanie obrazu dziala dla nazw <nazwa>#<i>.npy oraz <nazwa>#<i>#<ts>.npy.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image


def load_embeddings(emb_dir: Path, faces_dir: Path):
    files = sorted(emb_dir.glob("*.npy"))
    names, vectors, faces = [], [], []
    for f in files:
        parts = f.stem.split("#")
        face_path = faces_dir / f"{parts[0]}#twarz#{parts[1]}.jpg"
        if face_path.exists():
            img = np.array(Image.open(face_path).convert("RGB"))
        else:
            img = np.full((50, 50, 3), 200, dtype=np.uint8)
        names.append(f.stem)
        vectors.append(np.load(f))
        faces.append(img)
    X = np.array(vectors) if vectors else np.empty((0, 0))
    return names, X, faces


def load_brightness(emb_dir: Path, names: list) -> np.ndarray:
    table = {}
    meta = emb_dir / "meta.csv"
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
