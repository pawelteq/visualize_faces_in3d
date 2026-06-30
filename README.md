# Face Embeddings 3D

Wykrywa twarze na wielu zdjeciach, liczy embeddingi i pokazuje je w przestrzeni
**3D**. Cel: te same osoby laduja blisko siebie, bo o polozeniu decyduje
**twarz, a nie oswietlenie**. To wizualizacja, nie sztywny klasyfikator -
grupy widac wzrokowo (i opcjonalnie kolorem).

## Pipeline

```
images/                       zdjecia wejsciowe (duzo, rozne osoby)
   |  detect_faces.py         ArcFace (mocno wazy twarz) + CLAHE na swiatlo
   v
emb/<nazwa>#<i>.npy           embeddingi
emb/meta.csv                  jasnosc kadru (do tlumienia swiatla)
emb_images/...#twarz#<i>.jpg  podglady twarzy
   |  visualize_3d.py         L2-norm + tlumienie swiatla -> PCA/TSNE/LDA -> okno 3D
   v
okno 3D: twarze jako kafelki, kolor punktu = grupa (DBSCAN)
```

`group_faces.py` to opcjonalny dodatek - dorzuca etykiety grup do kolorowania
i do LDA, ale nie jest potrzebny do samej wizualizacji.

## Dlaczego twarz wazy mocniej niz swiatlo

1. **ArcFace** (domyslny backend) - embeddingi 512-D trenowane pod tozsamosc;
   z natury duzo mniej czule na swiatlo i poze niz stary dlib (128-D).
2. **CLAHE** w `detect_faces.py` - wyrownuje oswietlenie kadru przed embeddingiem.
3. **L2-normalizacja** - geometria kosinusowa (domyslnie wlaczona).
4. **Tlumienie swiatla** `--light-strength 0..1` - odejmuje liniowy wplyw jasnosci
   kadru z embeddingow. To pokretlo "o ile mocniej twarz niz swiatlo":
   `0` = surowo, `1` = pelne stlumienie swiatla (domyslnie).

> Porownanie efektu: uruchom raz z `--light-strength 1` i raz z `--light-strength 0`
> i zobacz, czy te same osoby skupiaja sie ciasniej.

## Instalacja

```bash
pip install -r requirements.txt
```

Domyslny backend ArcFace wymaga `insightface` + `onnxruntime` (pierwszy run
pobiera model). Jesli masz GPU, zainstaluj `onnxruntime-gpu` - przy duzej liczbie
zdjec mocno przyspiesza. Lzejsza alternatywa to `--backend dlib` (wtedy odkomentuj
`face_recognition` w requirements; na Windows wymaga CMake + VS Build Tools).

## Uzycie

```bash
# 1. Detekcja + embeddingi (ArcFace, z normalizacja swiatla)
python src/detect_faces.py
python src/detect_faces.py --backend dlib     # lzejszy wariant
python src/detect_faces.py --force            # przelicz od nowa

# 2. Wizualizacja 3D (domyslnie: twarz > swiatlo, kolor = grupa)
python src/visualize_3d.py
python src/visualize_3d.py --light-strength 0     # surowe embeddingi do porownania
python src/visualize_3d.py --reduce TSNE          # czesto lepsze skupiska niz PCA

# 3. (opcjonalnie) etykiety grup do kolorowania / LDA
python src/group_faces.py
python src/visualize_3d.py --reduce LDA --labels emb/labels_auto.csv --color-by label
```

## Moduly

```
src/
  detect_faces.py        # CLI: zdjecia -> embeddingi (+ meta.csv)
  visualize_3d.py        # CLI: embeddingi -> okno 3D
  group_faces.py         # CLI: grupowanie -> labels_auto.csv (opcjonalne)
  embeddings_backend.py  # backendy ArcFace / dlib (wspolny interfejs)
  preprocess.py          # CLAHE, jasnosc, wyciecie kadru
  postprocess.py         # L2-norm, tlumienie swiatla, LDA
  cluster.py             # DBSCAN, Chinese Whispers
  dataset.py             # wczytywanie embeddingow / meta / etykiet
```

## Uwagi o skali

- Przy bardzo duzej liczbie zdjec rozwaz `onnxruntime-gpu`.
- Nazwy plikow sa deterministyczne, wiec ponowne uruchomienie przetwarza tylko
  nowe zdjecia (stare embeddingi sa pomijane; `--force` liczy od nowa).
- t-SNE bywa wolne dla bardzo wielu punktow - do szybkiego podgladu uzyj PCA.
