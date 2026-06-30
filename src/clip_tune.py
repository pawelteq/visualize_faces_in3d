"""
Kalibracja filtra CLIP - bez ponownej detekcji.

Liczy wyniki "human" vs "reject" dla wycietych twarzy w podanym folderze
i pokazuje, ile zostaloby ODRZUCONYCH przy danym progu/marginesie oraz ktore
kadry sa najblizej granicy. Uruchom na folderze z PEWNYMI ludzkimi twarzami
(np. emb_images), zeby dobrac ustawienia, ktore ich NIE wycinaja.

Uzycie:
    python clip_tune.py --dir emb_images
    python clip_tune.py --dir emb_images --clip-threshold 0.12 --clip-margin 0.0 --limit 300
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from face_filter import ClipFaceFilter


def find_project_root(start: Path) -> Path:
    for c in (start, start.parent):
        if (c / "emb_images").exists():
            return c
    return start


ROOT = find_project_root(Path(__file__).resolve().parent)


def main() -> None:
    ap = argparse.ArgumentParser(description="Kalibracja filtra CLIP na folderze twarzy")
    ap.add_argument("--dir", default=str(ROOT / "emb_images"))
    ap.add_argument("--clip-threshold", type=float, default=0.15)
    ap.add_argument("--clip-margin", type=float, default=0.0)
    ap.add_argument("--clip-model", default="ViT-B-32")
    ap.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    ap.add_argument("--limit", type=int, default=300, help="Ile kadrow zbadac (0 = wszystkie)")
    ap.add_argument("--show", type=int, default=15, help="Ile granicznych przykladow wypisac")
    args = ap.parse_args()

    folder = Path(args.dir)
    crops = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        crops += list(folder.rglob(ext))
    crops = sorted(crops)
    if not crops:
        print(f"Brak obrazow w {folder}")
        return
    if args.limit and len(crops) > args.limit:
        step = len(crops) / args.limit
        crops = [crops[int(i * step)] for i in range(args.limit)]

    clip_device = "cpu" if args.device == "cpu" else "auto"
    flt = ClipFaceFilter(device=clip_device, threshold=args.clip_threshold,
                         margin=args.clip_margin, model_name=args.clip_model,
                         pretrained=args.clip_pretrained)

    rows = []
    for p in crops:
        try:
            crop = np.array(Image.open(p).convert("RGB"))
        except Exception:  # noqa: BLE001
            continue
        ok, h, r = flt.scores(crop)
        rows.append((p.name, ok, h, r, h - r))

    kept = sum(1 for _, ok, *_ in rows if ok)
    rej = len(rows) - kept
    print(f"\nZbadano {len(rows)} kadrow | prog={args.clip_threshold} margines={args.clip_margin}")
    print(f"  zostaloby: {kept}  | odrzucone: {rej} ({100*rej/max(1,len(rows)):.1f}%)\n")

    print(f"Najblizej granicy (najmniejsza przewaga human-reject):")
    for name, ok, h, r, d in sorted(rows, key=lambda t: t[4])[:args.show]:
        flag = "ODRZUC" if not ok else "ok"
        print(f"  {flag:6s} human={h:.3f} reject={r:.3f} (przewaga {d:+.3f})  {name}")
    print("\nJesli ODRZUCANE sa prawdziwe twarze - obniz --clip-threshold "
          "lub ustaw ujemny --clip-margin.")


if __name__ == "__main__":
    main()
