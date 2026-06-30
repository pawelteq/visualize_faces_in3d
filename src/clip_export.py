"""
Interaktywne strojenie filtra CLIP jako STRONA HTML.

CLIP liczy raz wyniki (human / reject) dla kadrow twarzy, a skrypt buduje
samodzielny clip_tuning.html z MINIATURAMI TYCH KADROW i suwakami. W przegladarce
przesuwasz prog/margines i na zywo widzisz, co zostaje, a co wypada.

Skad brac kadry (--source):
    images  : uruchom detektor na images/ i pokaz WSZYSTKIE kandydatury -
              twarze ORAZ falszywe (psy, grafiki). Najlepsze do strojenia,
              bo widzisz dokladnie to, co filtr ma odrzucic.
    crops   : uzyj juz wycietych twarzy z emb_images/ (domyslne).

Uzycie:
    python clip_export.py --source images --device gpu      # kandydatury z detekcji
    python clip_export.py --source crops --dir emb_images
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_project_root(start: Path) -> Path:
    for c in (start, start.parent):
        if (c / "emb_images").exists() or (c / "images").exists():
            return c
    return start


ROOT = find_project_root(Path(__file__).resolve().parent)


def make_thumb(crop_rgb: np.ndarray, size: int = 84) -> str:
    im = Image.fromarray(crop_rgb).convert("RGB")
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def sample_evenly(seq, limit):
    if not limit or len(seq) <= limit:
        return list(seq)
    step = len(seq) / limit
    return [seq[int(i * step)] for i in range(limit)]


def candidates_from_crops(folder: Path, limit: int):
    crops = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        crops += list(folder.rglob(ext))
    out = []
    for p in sample_evenly(sorted(crops), limit):
        try:
            out.append((p.name, np.array(Image.open(p).convert("RGB"))))
        except Exception:  # noqa: BLE001
            pass
    return out


def candidates_from_images(images_dir: Path, args):
    """Uruchamia detektor (bez filtra CLIP) i zwraca wszystkie wykryte kadry."""
    from embeddings_backend import get_backend
    backend = get_backend("arcface", normalize_light=True, min_score=args.min_score,
                          min_size=args.min_detect_size, device=args.device,
                          det_size=args.det_size)
    imgs = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]
    imgs = sample_evenly(imgs, args.max_images)
    out = []
    for n, p in enumerate(imgs, 1):
        try:
            image_rgb = np.array(Image.open(p).convert("RGB"))
        except Exception:  # noqa: BLE001
            continue
        for j, face in enumerate(backend.get_faces(image_rgb)):
            out.append((f"{p.stem}#{j}", face.crop))
        if n % 50 == 0:
            print(f"  detekcja {n}/{len(imgs)} | kandydatur: {len(out)}")
        if args.limit and len(out) >= args.limit:
            break
    return out


def build_html(items, threshold: float, margin: float) -> str:
    data = json.dumps(items)
    return """<!doctype html><html lang="pl"><head><meta charset="utf-8">
<title>Strojenie filtra CLIP</title>
<style>
 body{margin:0;font-family:system-ui,Arial,sans-serif;background:#15171c;color:#e8eaed}
 header{position:sticky;top:0;background:#1c1f26;padding:14px 18px;border-bottom:1px solid #2c303a;z-index:5}
 h1{font-size:17px;margin:0 0 10px}
 .ctl{display:flex;gap:26px;flex-wrap:wrap;align-items:center}
 .ctl label{font-size:13px;color:#aab}
 input[type=range]{width:240px;vertical-align:middle}
 .val{display:inline-block;min-width:48px;font-weight:600;color:#7fd1ff}
 .stats{margin-top:8px;font-size:14px}
 .stats b{color:#fff}
 .keep{color:#5fd28a}.rej{color:#ff7b7b}
 code{display:block;margin-top:8px;background:#0e1014;padding:8px 10px;border-radius:6px;
      font-size:12px;color:#bfe3ff;overflow:auto;white-space:pre}
 section{padding:10px 18px}
 h2{font-size:14px;margin:14px 0 8px;color:#cbd}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(94px,1fr));gap:8px}
 .cell{background:#1c1f26;border:1px solid #2c303a;border-radius:8px;padding:5px;text-align:center}
 .cell img{width:84px;height:84px;object-fit:cover;border-radius:5px;display:block;margin:0 auto}
 .cell .s{font-size:10px;color:#99a;margin-top:3px;line-height:1.25}
 .cell.r{border-color:#5a2a2a}
 .muted{color:#778}
</style></head><body>
<header>
 <h1>Strojenie filtra CLIP — co zostaje, a co wypada</h1>
 <div class="ctl">
  <div><label>prog (human &ge;)</label><br><input id="th" type="range" min="0" max="0.40" step="0.005">
       <span class="val" id="thv"></span></div>
  <div><label>margines (human-reject &ge;)</label><br><input id="mg" type="range" min="-0.10" max="0.15" step="0.005">
       <span class="val" id="mgv"></span></div>
  <div class="stats">Zostaje: <b class="keep" id="nk"></b> &nbsp; Wypada: <b class="rej" id="nr"></b>
       <span class="muted" id="pct"></span></div>
 </div>
 <code id="cmd"></code>
</header>
<section>
 <h2 class="rej">Wypada (sortowane: najblizej granicy u gory — tu szukaj prawdziwych twarzy)</h2>
 <div class="grid" id="gridR"></div>
 <h2 class="keep">Zostaje</h2>
 <div class="grid" id="gridK"></div>
</section>
<script>
const DATA=__DATA__;
const th=document.getElementById('th'), mg=document.getElementById('mg');
th.value=__TH__; mg.value=__MG__;
const thv=document.getElementById('thv'),mgv=document.getElementById('mgv');
const nk=document.getElementById('nk'),nr=document.getElementById('nr'),pct=document.getElementById('pct');
const gridK=document.getElementById('gridK'),gridR=document.getElementById('gridR'),cmd=document.getElementById('cmd');
function cell(it,rej){return `<div class="cell ${rej?'r':''}"><img src="${it.img}" loading="lazy">`
  +`<div class="s">h ${it.h.toFixed(3)}<br>r ${it.r.toFixed(3)}<br>${(it.h-it.r>=0?'+':'')}${(it.h-it.r).toFixed(3)}</div></div>`;}
function render(){
  const T=+th.value,M=+mg.value; thv.textContent=T.toFixed(3); mgv.textContent=M.toFixed(3);
  let keep=[],rej=[];
  for(const it of DATA){ (it.h>=T && (it.h-it.r)>=M ? keep:rej).push(it); }
  rej.sort((a,b)=>(b.h-b.r)-(a.h-a.r));
  keep.sort((a,b)=>(a.h-a.r)-(b.h-b.r));
  nk.textContent=keep.length; nr.textContent=rej.length;
  pct.textContent=` (z ${DATA.length}, wypada ${(100*rej.length/Math.max(1,DATA.length)).toFixed(1)}%)`;
  gridR.innerHTML=rej.map(it=>cell(it,true)).join('');
  gridK.innerHTML=keep.map(it=>cell(it,false)).join('');
  cmd.textContent=`python src/detect_faces.py --device gpu --rebuild --min-score 0.7 --clip-filter `
    +`--clip-threshold ${T.toFixed(3)} --clip-margin ${M.toFixed(3)}`;
}
th.oninput=render; mg.oninput=render; render();
</script></body></html>""".replace("__DATA__", data).replace("__TH__", repr(threshold)).replace("__MG__", repr(margin))


def main() -> None:
    ap = argparse.ArgumentParser(description="Strojenie filtra CLIP jako strona HTML")
    ap.add_argument("--source", choices=["images", "crops"], default="images",
                    help="images = kandydatury z detekcji (twarze + psy/grafiki); crops = emb_images")
    ap.add_argument("--images", default=str(ROOT / "images"))
    ap.add_argument("--dir", default=str(ROOT / "emb_images"))
    ap.add_argument("--out", default=str(ROOT / "clip_tuning.html"))
    ap.add_argument("--limit", type=int, default=500, help="Max kadrow na stronie (0 = bez limitu)")
    ap.add_argument("--max-images", type=int, default=600, help="Ile zdjec przeszukac (source=images)")
    ap.add_argument("--thumb", type=int, default=84)
    ap.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--min-score", type=float, default=0.6)
    ap.add_argument("--min-detect-size", type=int, default=40)
    ap.add_argument("--clip-threshold", type=float, default=0.14)
    ap.add_argument("--clip-margin", type=float, default=0.0)
    ap.add_argument("--clip-model", default="ViT-B-32")
    ap.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    args = ap.parse_args()

    if args.source == "images":
        print(f"Detekcja kandydatur z {args.images} ...")
        candidates = candidates_from_images(Path(args.images), args)
    else:
        candidates = candidates_from_crops(Path(args.dir), args.limit)
    if not candidates:
        print("Brak kadrow do pokazania.")
        return
    print(f"Kadrow do oceny: {len(candidates)}. Licze CLIP...")

    from face_filter import ClipFaceFilter
    clip_device = "cpu" if args.device == "cpu" else "auto"
    flt = ClipFaceFilter(device=clip_device, threshold=args.clip_threshold,
                         margin=args.clip_margin, model_name=args.clip_model,
                         pretrained=args.clip_pretrained)

    items = []
    for n, (name, crop) in enumerate(candidates, 1):
        _ok, h, r = flt.scores(crop)
        items.append({"name": name, "img": make_thumb(crop, args.thumb),
                      "h": round(float(h), 4), "r": round(float(r), 4)})
        if n % 100 == 0:
            print(f"  CLIP {n}/{len(candidates)}")

    html = build_html(items, args.clip_threshold, args.clip_margin)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"\nGotowe: {args.out}  ({len(items)} kadrow). Otworz w przegladarce.")


if __name__ == "__main__":
    main()
