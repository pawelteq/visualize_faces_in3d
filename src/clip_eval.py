"""
Test (ewaluacja) filtra CLIP - szybkie strojenie liczbami zamiast na oko.

Dajesz dwa zbiory przykladow:
    - TWARZE      (prawdziwe ludzkie twarze),
    - NIE-TWARZE  (psy, rendery, roboty, grafiki, ...).
Skrypt liczy CLIP RAZ dla kazdego kadru, przeszukuje wszystkie progi i marginesy
i znajduje ustawienia o najlepszym wyniku (F1). Wypisuje macierz pomylek oraz
co jeszcze MYLI (twarze blednie odrzucone i smieci blednie zostawione) - to
podpowiada, jakie opisy dodac w face_filter.py.

Skad brac przyklady:
    --auto                 TWARZE = foldery osob w emb_images/, NIE-TWARZE = foldery
                           nierozpoznane/zwierzeta (DEFAULT, jesli masz juz tak ulozone),
    --faces-dir D1 --not-faces-dir D2   wlasne foldery z przykladami.

Uzycie:
    python clip_eval.py --auto --device gpu
    python clip_eval.py --faces-dir eval/face --not-faces-dir eval/junk --html
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import DEFAULT_IGNORE
from clip_export import make_thumb


def find_project_root(start: Path) -> Path:
    for c in (start, start.parent):
        if (c / "emb_images").exists():
            return c
    return start


ROOT = find_project_root(Path(__file__).resolve().parent)
EXTS = ("*.jpg", "*.jpeg", "*.png")


def gather(folder: Path, limit: int):
    files = []
    for ext in EXTS:
        files += list(folder.rglob(ext))
    files = sorted(files)
    if limit and len(files) > limit:
        step = len(files) / limit
        files = [files[int(i * step)] for i in range(limit)]
    return files


def gather_auto(faces_dir: Path, limit: int):
    pos, neg = [], []
    for sub in sorted(p for p in faces_dir.iterdir() if p.is_dir()):
        target = neg if sub.name.strip().lower() in DEFAULT_IGNORE else pos
        for ext in EXTS:
            target += list(sub.rglob(ext))
    def cap(lst):
        lst = sorted(lst)
        if limit and len(lst) > limit:
            step = len(lst) / limit
            lst = [lst[int(i * step)] for i in range(limit)]
        return lst
    return cap(pos), cap(neg)


def grid_search(H, R, y, prefer="f1"):
    """Przeszukuje prog i margines. y=1 dla TWARZY. Zwraca dict z najlepszymi."""
    ths = np.round(np.arange(0.0, 0.401, 0.01), 3)
    mgs = np.round(np.arange(-0.10, 0.151, 0.01), 3)
    y = np.asarray(y, dtype=bool)
    best = None
    for t in ths:
        for m in mgs:
            keep = (H >= t) & (H - R >= m)
            tp = int(np.sum(keep & y))
            fp = int(np.sum(keep & ~y))
            fn = int(np.sum(~keep & y))
            tn = int(np.sum(~keep & ~y))
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            bal = 0.5 * (rec + (tn / (tn + fp) if tn + fp else 0.0))
            score = {"f1": f1, "balanced": bal, "recall": rec, "precision": prec}[prefer]
            cand = dict(t=float(t), m=float(m), tp=tp, fp=fp, fn=fn, tn=tn,
                        precision=prec, recall=rec, f1=f1, balanced=bal, score=score)
            if best is None or cand["score"] > best["score"]:
                best = cand
    return best


def write_mistakes_html(out: Path, fp_items, fn_items):
    def grid(items):
        return "".join(
            f'<div class=c><img src="{it["img"]}"><div>h {it["h"]:.3f} r {it["r"]:.3f}<br>{it["name"]}</div></div>'
            for it in items)
    html = f"""<!doctype html><meta charset=utf-8><title>Pomylki filtra CLIP</title>
<style>body{{background:#15171c;color:#e8eaed;font-family:system-ui,Arial}}
h2{{padding:0 14px}}.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:8px;padding:0 14px}}
.c{{background:#1c1f26;border:1px solid #2c303a;border-radius:8px;padding:5px;text-align:center;font-size:10px;color:#9aa}}
.c img{{width:96px;height:96px;object-fit:cover;border-radius:5px}}</style>
<h2 style="color:#ff7b7b">SMIECI blednie zostawione ({len(fp_items)}) — dodaj dla nich opisy 'reject'</h2>
<div class=g>{grid(fp_items)}</div>
<h2 style="color:#5fd28a">TWARZE blednie odrzucone ({len(fn_items)}) — za ostro</h2>
<div class=g>{grid(fn_items)}</div>"""
    out.write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ewaluacja/strojenie filtra CLIP")
    ap.add_argument("--auto", action="store_true",
                    help="TWARZE = foldery osob w emb_images, NIE-TWARZE = nierozpoznane/zwierzeta")
    ap.add_argument("--faces-dir", default="")
    ap.add_argument("--not-faces-dir", default="")
    ap.add_argument("--emb-images", default=str(ROOT / "emb_images"))
    ap.add_argument("--limit", type=int, default=300, help="Max przykladow na klase")
    ap.add_argument("--prefer", choices=["f1", "balanced", "recall", "precision"], default="f1")
    ap.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    ap.add_argument("--clip-model", default="ViT-B-32")
    ap.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--html", action="store_true", help="Zapisz clip_eval.html z pomylkami")
    ap.add_argument("--out", default=str(ROOT / "clip_eval.html"))
    args = ap.parse_args()

    if args.faces_dir and args.not_faces_dir:
        pos = gather(Path(args.faces_dir), args.limit)
        neg = gather(Path(args.not_faces_dir), args.limit)
    else:
        if not args.auto:
            print("Podaj --auto albo --faces-dir i --not-faces-dir.")
            return
        pos, neg = gather_auto(Path(args.emb_images), args.limit)

    if len(pos) < 3 or len(neg) < 3:
        print(f"Za malo przykladow (twarze={len(pos)}, nie-twarze={len(neg)}). "
              "Potrzeba po kilka/kilkanascie z kazdej klasy.")
        return
    print(f"Przyklady: TWARZE={len(pos)} | NIE-TWARZE={len(neg)}. Licze CLIP...")

    from face_filter import ClipFaceFilter
    clip_device = "cpu" if args.device == "cpu" else "auto"
    flt = ClipFaceFilter(device=clip_device, model_name=args.clip_model,
                         pretrained=args.clip_pretrained)

    rows = []  # (path, y, h, r)
    for p in pos:
        _o, h, r = flt.scores(np.array(Image.open(p).convert("RGB")))
        rows.append((p, 1, h, r))
    for p in neg:
        _o, h, r = flt.scores(np.array(Image.open(p).convert("RGB")))
        rows.append((p, 0, h, r))

    H = np.array([r[2] for r in rows])
    R = np.array([r[3] for r in rows])
    y = np.array([r[1] for r in rows])

    best = grid_search(H, R, y, prefer=args.prefer)
    n_pos, n_neg = int(np.sum(y)), int(np.sum(1 - y))
    print("\n=== NAJLEPSZE USTAWIENIA (wg " + args.prefer + ") ===")
    print(f"  --clip-threshold {best['t']:.2f} --clip-margin {best['m']:.2f}")
    print(f"  F1={best['f1']:.3f}  precyzja={best['precision']:.3f}  czulosc={best['recall']:.3f}  "
          f"balanced_acc={best['balanced']:.3f}")
    print(f"  Twarze: zachowane {best['tp']}/{n_pos}, zgubione {best['fn']}")
    print(f"  Nie-twarze: odrzucone {best['tn']}/{n_neg}, przepuszczone {best['fp']}")
    print(f"\nUzyj:  python src/detect_faces.py --device gpu --rebuild --min-score 0.7 "
          f"--clip-filter --clip-threshold {best['t']:.2f} --clip-margin {best['m']:.2f}")

    # pomylki przy najlepszych ustawieniach
    t, m = best["t"], best["m"]
    keep = (H >= t) & (H - R >= m)
    fp_idx = [i for i in range(len(rows)) if keep[i] and y[i] == 0]
    fn_idx = [i for i in range(len(rows)) if not keep[i] and y[i] == 1]
    fp_idx.sort(key=lambda i: -(H[i] - R[i]))
    fn_idx.sort(key=lambda i: (H[i] - R[i]))

    print(f"\nSMIECI przepuszczone ({len(fp_idx)}) - dla nich warto dodac opisy 'reject':")
    for i in fp_idx[:12]:
        print(f"   h={H[i]:.3f} r={R[i]:.3f}  {rows[i][0].name}")
    print(f"TWARZE zgubione ({len(fn_idx)}):")
    for i in fn_idx[:12]:
        print(f"   h={H[i]:.3f} r={R[i]:.3f}  {rows[i][0].name}")

    if args.html:
        def item(i):
            return {"img": make_thumb(np.array(Image.open(rows[i][0]).convert("RGB"))),
                    "h": float(H[i]), "r": float(R[i]), "name": rows[i][0].name}
        write_mistakes_html(Path(args.out), [item(i) for i in fp_idx], [item(i) for i in fn_idx])
        print(f"\nZapisano pomylki do podgladu: {args.out}")


if __name__ == "__main__":
    main()
