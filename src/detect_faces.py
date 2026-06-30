"""
Detekcja twarzy + generowanie embeddingow.

Czyta zdjecia z images/, wykrywa twarze wybranym backendem i zapisuje:
    emb/<nazwa>#<i>.npy                embedding
    emb_images/<nazwa>#twarz#<i>.jpg   podglad twarzy
    emb/meta.csv                       jasnosc kadru + backend (kowiata do regresji swiatla)

Nazwy sa deterministyczne -> ponowne uruchomienie nie tworzy duplikatow
(istniejace embeddingi sa pomijane; --force przelicza od nowa).

Przyklady:
    python detect_faces.py
    python detect_faces.py --backend dlib
    python detect_faces.py --no-light-norm
    python detect_faces.py --force
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from embeddings_backend import get_backend

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_project_root(start: Path) -> Path:
    for candidate in (start, start.parent):
        if (candidate / "images").exists() or (candidate / "emb").exists():
            return candidate
    return start


ROOT = find_project_root(Path(__file__).resolve().parent)


def list_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        print(f"Folder ze zdjeciami nie istnieje: {images_dir}")
        return []
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def load_meta(meta_path: Path) -> dict:
    rows = {}
    if meta_path.exists():
        with open(meta_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows[row["name"]] = row
    return rows


def write_meta(meta_path: Path, rows: dict) -> None:
    with open(meta_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "brightness", "backend"])
        writer.writeheader()
        for name in sorted(rows):
            writer.writerow(rows[name])


def main() -> None:
    parser = argparse.ArgumentParser(description="Detekcja twarzy -> embeddingi")
    parser.add_argument("--backend", choices=["dlib", "arcface"], default="arcface",
                        help="arcface: mocniej wazy twarz niz swiatlo (zalecane); dlib: lzejszy")
    parser.add_argument("--images", default=str(ROOT / "images"))
    parser.add_argument("--emb", default=str(ROOT / "emb"))
    parser.add_argument("--faces", default=str(ROOT / "emb_images"))
    parser.add_argument("--no-light-norm", action="store_true",
                        help="Wylacz normalizacje oswietlenia (CLAHE)")
    parser.add_argument("--force", action="store_true",
                        help="Przelicz nawet gdy embedding juz istnieje")
    args = parser.parse_args()

    images_dir, emb_dir, faces_dir = Path(args.images), Path(args.emb), Path(args.faces)
    emb_dir.mkdir(parents=True, exist_ok=True)
    faces_dir.mkdir(parents=True, exist_ok=True)

    backend = get_backend(args.backend, normalize_light=not args.no_light_norm)
    meta = load_meta(emb_dir / "meta.csv")

    images = list_images(images_dir)
    if not images:
        print("Brak zdjec do przetworzenia.")
        return

    print(f"Backend: {backend.name} | normalizacja swiatla: {not args.no_light_norm}")
    print(f"Znaleziono {len(images)} zdjec w {images_dir}")

    total = 0
    for path in images:
        base = path.stem
        try:
            image_rgb = np.array(Image.open(path).convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            print(f"  Nie udalo sie otworzyc {path.name}: {exc}")
            continue

        faces = backend.get_faces(image_rgb)
        if not faces:
            print(f"  {path.name}: brak twarzy")
            continue

        new_here = 0
        for i, face in enumerate(faces):
            name = f"{base}#{i}"
            emb_path = emb_dir / f"{name}.npy"
            if emb_path.exists() and not args.force:
                continue
            np.save(emb_path, face.embedding)
            Image.fromarray(face.crop).save(faces_dir / f"{base}#twarz#{i}.jpg")
            meta[name] = {"name": name,
                          "brightness": f"{face.brightness:.3f}",
                          "backend": backend.name}
            new_here += 1
            total += 1
        print(f"  {path.name}: {len(faces)} twarz(y), nowych: {new_here}")

    write_meta(emb_dir / "meta.csv", meta)
    print(f"Gotowe. Nowych embeddingow lacznie: {total}")


if __name__ == "__main__":
    main()
