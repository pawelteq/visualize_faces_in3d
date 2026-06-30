"""
Detekcja twarzy + generowanie embeddingow.

Czyta zdjecia z images/, wykrywa twarze wybranym backendem i zapisuje:
    emb/<nazwa>#<i>.npy                embedding
    emb_images/<nazwa>#twarz#<i>.jpg   podglad twarzy
    emb/meta.csv                       jasnosc + pewnosc detekcji + backend

Wydajnosc (duzo zdjec):
    --device gpu / auto / cpu          GPU (CUDA) jest wielokrotnie szybsze,
    --det-size N                       rozdzielczosc detektora (mniejsza = szybciej).

Mniej falszywych twarzy:
    --min-score N        odrzuc detekcje ponizej pewnosci N (ArcFace; domyslnie 0.6),
    --min-detect-size N  odrzuc ramki mniejsze niz N px (domyslnie 40),
    --clip-filter        ODRZUCA psy/zwierzeta/grafiki (semantyczny filtr CLIP;
                         pip install open_clip_torch torch). Decyzja wzgledna,
                         wiec twarze pod katem przechodza. Stroj: --clip-threshold,
                         --clip-margin (ujemny = lagodniej), --clip-model.

Przeliczanie:
    --force              nadpisz istniejace embeddingi,
    --rebuild            wyczysc emb/ i emb_images/ (z folderami osob!) i wykryj od nowa
                         (pyta o potwierdzenie, chyba ze --yes).

Przyklady:
    python detect_faces.py --device gpu --rebuild --min-score 0.7 --clip-filter
    python detect_faces.py --device gpu --clip-filter --clip-threshold 0.12 --clip-margin -0.02
"""
from __future__ import annotations

import argparse
import csv
import shutil
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


def wipe_dir(d: Path) -> int:
    n = 0
    if d.exists():
        for item in d.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            n += 1
    return n


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
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--images", default=str(ROOT / "images"))
    parser.add_argument("--emb", default=str(ROOT / "emb"))
    parser.add_argument("--faces", default=str(ROOT / "emb_images"))
    parser.add_argument("--min-score", type=float, default=0.6)
    parser.add_argument("--min-detect-size", type=int, default=40)
    parser.add_argument("--dlib-model", choices=["hog", "cnn"], default="hog")
    parser.add_argument("--clip-filter", action="store_true",
                        help="Odrzuc psy/zwierzeta/grafiki filtrem CLIP (wymaga open_clip_torch)")
    parser.add_argument("--clip-threshold", type=float, default=0.14)
    parser.add_argument("--clip-margin", type=float, default=0.0,
                        help="Wymagana przewaga 'twarzy' nad 'nie-twarza' (ujemny = lagodniej)")
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--no-light-norm", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rebuild", action="store_true",
                        help="Wyczysc emb/ i emb_images/ (z folderami osob!) i wykryj od nowa")
    parser.add_argument("--yes", action="store_true", help="Nie pytaj o potwierdzenie --rebuild")
    args = parser.parse_args()

    images_dir, emb_dir, faces_dir = Path(args.images), Path(args.emb), Path(args.faces)

    if args.rebuild:
        print(f"REBUILD: usune CALA zawartosc:\n  {emb_dir}\n  {faces_dir}")
        print("  (w tym recznie poukladane foldery osob - etykiety bedzie trzeba zrobic od nowa)")
        if not args.yes:
            ans = input("Kontynuowac? wpisz 'tak': ").strip().lower()
            if ans not in ("tak", "t", "yes", "y"):
                print("Anulowano.")
                return
        removed = wipe_dir(emb_dir) + wipe_dir(faces_dir)
        print(f"Wyczyszczono ({removed} elementow).")
        args.force = True

    emb_dir.mkdir(parents=True, exist_ok=True)
    faces_dir.mkdir(parents=True, exist_ok=True)

    backend = get_backend(args.backend, normalize_light=not args.no_light_norm,
                          min_score=args.min_score, min_size=args.min_detect_size,
                          dlib_model=args.dlib_model, device=args.device, det_size=args.det_size)

    clip_filter = None
    if args.clip_filter:
        from face_filter import ClipFaceFilter
        clip_device = "cpu" if args.device == "cpu" else "auto"
        clip_filter = ClipFaceFilter(device=clip_device, threshold=args.clip_threshold,
                                     margin=args.clip_margin, model_name=args.clip_model,
                                     pretrained=args.clip_pretrained)

    meta = load_meta(emb_dir / "meta.csv")

    images = list_images(images_dir)
    if not images:
        print("Brak zdjec do przetworzenia.")
        return

    print(f"Backend: {backend.name} | min_score={args.min_score} | min_size={args.min_detect_size}px"
          f"{' | CLIP-filtr ON' if clip_filter else ''}")
    print(f"Znaleziono {len(images)} zdjec w {images_dir}\n")

    total, rejected_total, clip_rejected = 0, 0, 0
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

        if clip_filter is not None:
            kept = []
            for face in faces:
                ok, _h, _r = clip_filter.is_human_face(face.crop)
                if ok:
                    kept.append(face)
                else:
                    clip_rejected += 1
            faces = kept

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
          f"odrzucono detektorem: {rejected_total}"
          f"{f' | odrzucono CLIP (psy/grafiki): {clip_rejected}' if clip_filter else ''}")


if __name__ == "__main__":
    main()
