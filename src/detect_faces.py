"""
Detekcja twarzy + generowanie embeddingow.

Czyta zdjecia z images/, wykrywa twarze wybranym backendem i zapisuje:
    emb/<nazwa>#<i>.npy                embedding
    emb_images/<nazwa>#twarz#<i>.jpg   podglad twarzy
    emb/meta.csv                       jasnosc + pewnosc detekcji + backend

Wydajnosc (duzo zdjec):
    --device gpu        wymus GPU (CUDA) - wymaga onnxruntime-gpu,
    --device auto       GPU jesli dostepne, inaczej CPU (domyslnie),
    --det-size N        rozdzielczosc detektora (mniejsza = szybciej, gubi male twarze).

Mniej falszywych twarzy (szumy, zwierzeta):
    --min-score N       odrzuc detekcje ponizej pewnosci N (ArcFace; domyslnie 0.6),
    --min-detect-size N odrzuc ramki mniejsze niz N px (domyslnie 40),
    --dlib-model cnn    dokladniejszy detektor dla backendu dlib.

Przyklady:
    python detect_faces.py --device gpu
    python detect_faces.py --device gpu --det-size 480 --min-score 0.7
"""
from __future__ import annotations

import argparse
import csv
import time
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
    fields = ["name", "brightness", "score", "backend"]
    with open(meta_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for name in sorted(rows):
            writer.writerow({k: rows[name].get(k, "") for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Detekcja twarzy -> embeddingi")
    parser.add_argument("--backend", choices=["dlib", "arcface"], default="arcface")
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto",
                        help="auto = GPU jesli dostepne; gpu = wymus CUDA; cpu = wymus CPU")
    parser.add_argument("--det-size", type=int, default=640,
                        help="Rozdzielczosc detektora ArcFace (mniejsza = szybciej)")
    parser.add_argument("--images", default=str(ROOT / "images"))
    parser.add_argument("--emb", default=str(ROOT / "emb"))
    parser.add_argument("--faces", default=str(ROOT / "emb_images"))
    parser.add_argument("--min-score", type=float, default=0.6)
    parser.add_argument("--min-detect-size", type=int, default=40)
    parser.add_argument("--dlib-model", choices=["hog", "cnn"], default="hog")
    parser.add_argument("--no-light-norm", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    images_dir, emb_dir, faces_dir = Path(args.images), Path(args.emb), Path(args.faces)
    emb_dir.mkdir(parents=True, exist_ok=True)
    faces_dir.mkdir(parents=True, exist_ok=True)

    backend = get_backend(args.backend, normalize_light=not args.no_light_norm,
                          min_score=args.min_score, min_size=args.min_detect_size,
                          dlib_model=args.dlib_model, device=args.device, det_size=args.det_size)
    meta = load_meta(emb_dir / "meta.csv")

    images = list_images(images_dir)
    if not images:
        print("Brak zdjec do przetworzenia.")
        return

    print(f"Backend: {backend.name} | min_score={args.min_score} | min_size={args.min_detect_size}px")
    print(f"Znaleziono {len(images)} zdjec w {images_dir}\n")

    total, rejected_total = 0, 0
    t0 = time.time()
    for n, path in enumerate(images, 1):
        base = path.stem
        try:
            image_rgb = np.array(Image.open(path).convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            print(f"  Nie udalo sie otworzyc {path.name}: {exc}")
            continue

        faces = backend.get_faces(image_rgb)
        rejected_total += getattr(backend, "last_rejected", 0)
        for i, face in enumerate(faces):
            name = f"{base}#{i}"
            emb_path = emb_dir / f"{name}.npy"
            if emb_path.exists() and not args.force:
                continue
            np.save(emb_path, face.embedding)
            Image.fromarray(face.crop).save(faces_dir / f"{base}#twarz#{i}.jpg")
            meta[name] = {"name": name, "brightness": f"{face.brightness:.3f}",
                          "score": f"{face.score:.3f}", "backend": backend.name}
            total += 1

        if n % 50 == 0 or n == len(images):
            rate = n / (time.time() - t0 + 1e-9)
            print(f"  {n}/{len(images)} zdjec | {rate:.1f} zdj/s | embeddingow: {total}")

    write_meta(emb_dir / "meta.csv", meta)
    dt = time.time() - t0
    print(f"\nGotowe w {dt:.1f}s. Nowych embeddingow: {total} | "
          f"odrzucono (niska pewnosc/maly rozmiar): {rejected_total}")


if __name__ == "__main__":
    main()
