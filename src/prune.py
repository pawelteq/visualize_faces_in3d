"""
Czyszczenie istniejacego zbioru z detekcji o niskiej pewnosci (szumy, zwierzeta).

Czyta emb/meta.csv i usuwa embeddingi (oraz ich podglady twarzy), ktorych
pewnosc detekcji jest ponizej progu. Dziala tylko na danych, ktore maja zapisany
'score' w meta.csv (czyli wygenerowanych nowym detect_faces.py).

Domyslnie BEZPIECZNIE: tylko pokazuje, co usunie. Dodaj --apply, aby usunac.

Przyklady:
    python prune.py --min-score 0.7            # podglad
    python prune.py --min-score 0.7 --apply    # usun naprawde
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from dataset import _index_faces, _face_key


def find_project_root(start: Path) -> Path:
    for c in (start, start.parent):
        if (c / "emb").exists():
            return c
    return start


ROOT = find_project_root(Path(__file__).resolve().parent)


def main() -> None:
    ap = argparse.ArgumentParser(description="Usuwanie detekcji o niskiej pewnosci")
    ap.add_argument("--emb", default=str(ROOT / "emb"))
    ap.add_argument("--faces", default=str(ROOT / "emb_images"))
    ap.add_argument("--min-score", type=float, default=0.6)
    ap.add_argument("--apply", action="store_true", help="Faktycznie usun (domyslnie tylko podglad)")
    args = ap.parse_args()

    emb_dir, faces_dir = Path(args.emb), Path(args.faces)
    meta_path = emb_dir / "meta.csv"
    if not meta_path.exists():
        print(f"Brak {meta_path} - uruchom najpierw detect_faces.py (zapisuje 'score').")
        return

    path_index, _ = _index_faces(faces_dir)
    to_remove = []
    no_score = 0
    with open(meta_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw = row.get("score", "")
            if raw == "":
                no_score += 1
                continue
            if float(raw) < args.min_score:
                to_remove.append((row["name"], float(raw)))

    print(f"Do usuniecia (score < {args.min_score}): {len(to_remove)} detekcji"
          f"{f' | bez zapisanego score (pomijam): {no_score}' if no_score else ''}")
    for name, sc in sorted(to_remove, key=lambda t: t[1])[:20]:
        print(f"  {name}  (score={sc:.2f})")
    if len(to_remove) > 20:
        print(f"  ... i {len(to_remove) - 20} wiecej")

    if not args.apply:
        print("\nTo byl tylko podglad. Dodaj --apply, aby usunac.")
        return

    removed = 0
    for name, _sc in to_remove:
        npy = emb_dir / f"{name}.npy"
        if npy.exists():
            npy.unlink()
            removed += 1
        crop = path_index.get(_face_key(name))
        if crop is not None and Path(crop).exists():
            Path(crop).unlink()
    print(f"Usunieto {removed} embeddingow (+ pasujace podglady twarzy).")
    print("Wskazowka: usun te wpisy z meta.csv ponownie uruchamiajac detect_faces.py, "
          "albo zignoruj - brakujace pliki sa pomijane przy wczytywaniu.")


if __name__ == "__main__":
    main()
