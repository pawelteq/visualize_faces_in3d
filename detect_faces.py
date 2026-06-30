"""
Detekcja twarzy i generowanie embeddingow.

Czyta zdjecia z folderu images/, wykrywa twarze (face_recognition / dlib),
i zapisuje:
    - 128-wymiarowe embeddingi  -> emb/<nazwa>#<i>.npy
    - wyciete twarze (podglad)  -> emb_images/<nazwa>#twarz#<i>.jpg

Nazwy plikow sa DETERMINISTYCZNE (bez znacznika czasu), wiec ponowne
uruchomienie nie tworzy duplikatow - istniejace embeddingi sa pomijane.
Uzyj --force, aby przeliczyc wszystko od nowa.

Przyklady:
    python detect_faces.py
    python detect_faces.py --force
    python detect_faces.py --images sciezka/do/zdjec
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import face_recognition
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_project_root(start: Path) -> Path:
    """Zwraca katalog projektu niezaleznie od tego, czy skrypt lezy w src/, czy w korzeniu."""
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


def process_image(path: Path, emb_dir: Path, faces_dir: Path, force: bool = False) -> int:
    """Przetwarza jedno zdjecie. Zwraca liczbe nowo zapisanych embeddingow."""
    base = path.stem

    try:
        image = face_recognition.load_image_file(str(path))
    except Exception as exc:  # noqa: BLE001
        print(f"  Nie udalo sie otworzyc {path.name}: {exc}")
        return 0

    locations = face_recognition.face_locations(image)
    if not locations:
        print(f"  {path.name}: brak twarzy")
        return 0

    encodings = face_recognition.face_encodings(image, locations)
    pil = Image.fromarray(image)

    saved = 0
    for i, (encoding, (top, right, bottom, left)) in enumerate(zip(encodings, locations)):
        emb_path = emb_dir / f"{base}#{i}.npy"
        face_path = faces_dir / f"{base}#twarz#{i}.jpg"

        if emb_path.exists() and not force:
            continue

        np.save(emb_path, encoding)
        pil.crop((left, top, right, bottom)).save(face_path)
        saved += 1

    print(f"  {path.name}: {len(locations)} twarz(y), nowych embeddingow: {saved}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Detekcja twarzy -> embeddingi")
    parser.add_argument("--images", default=str(ROOT / "images"), help="Folder ze zdjeciami wejsciowymi")
    parser.add_argument("--emb", default=str(ROOT / "emb"), help="Folder na embeddingi .npy")
    parser.add_argument("--faces", default=str(ROOT / "emb_images"), help="Folder na wyciete twarze")
    parser.add_argument("--force", action="store_true", help="Przelicz nawet gdy embedding juz istnieje")
    args = parser.parse_args()

    images_dir = Path(args.images)
    emb_dir = Path(args.emb)
    faces_dir = Path(args.faces)
    emb_dir.mkdir(parents=True, exist_ok=True)
    faces_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(images_dir)
    if not images:
        print("Brak zdjec do przetworzenia.")
        return

    print(f"Znaleziono {len(images)} zdjec w {images_dir}")
    total = sum(process_image(p, emb_dir, faces_dir, force=args.force) for p in images)
    print(f"Gotowe. Nowych embeddingow lacznie: {total}")


if __name__ == "__main__":
    main()
