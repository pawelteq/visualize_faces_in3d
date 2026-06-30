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

Domyslny backend ArcFace wymaga `insightface` + `onnxruntime-gpu` (pierwszy run
pobiera model). Lzejsza alternatywa to `--backend dlib` (wtedy odkomentuj
`face_recognition` w requirements; na Windows wymaga CMake + VS Build Tools).

### GPU (duzo zdjec)

ArcFace liczy sie na GPU (CUDA), co przy tysiacach zdjec jest wielokrotnie
szybsze niz CPU. Potrzebujesz:

- karty NVIDIA + sterownikow CUDA i cuDNN,
- pakietu `onnxruntime-gpu` (zamiast `onnxruntime`).

Uruchomienie:

```bash
python src/detect_faces.py --device gpu              # wymus GPU (blad, jesli niedostepne)
python src/detect_faces.py --device auto             # GPU jesli jest, inaczej CPU
python src/detect_faces.py --device gpu --det-size 480   # szybciej (mniejsza detekcja)
```

Skrypt wypisuje, na czym liczy (`ArcFace na: GPU (CUDA)`) oraz tempo
(`zdj/s`), wiec od razu widac, czy GPU faktycznie pracuje. Sprawdza realnie
zaladowane providery - jesli CUDA cicho spadnie na CPU, dostaniesz ostrzezenie.

**Czesty blad: `cublasLt64_12.dll missing`** (CUDA cicho spada na CPU).
Oznacza brak bibliotek CUDA 12. Najprostsza naprawa (skrypt sam je znajdzie):

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12
```

Sprawdz tez, czy `onnxruntime-gpu` pasuje do CUDA 12 (`pip show onnxruntime-gpu`;
nowsze wersje = CUDA 12 + cuDNN 9). Alternatywa: NVIDIA CUDA Toolkit 12 + cuDNN 9
zainstalowane systemowo, z `\bin` na PATH.

## Uzycie

```bash
# 1. Detekcja + embeddingi (ArcFace, z normalizacja swiatla)
python src/detect_faces.py --device gpu       # GPU (CUDA) - szybko przy wielu zdjeciach
python src/detect_faces.py --min-score 0.7    # mniej szumow/zwierzat (ostrzejszy prog)
python src/detect_faces.py --backend dlib --dlib-model cnn   # lzejszy, dokladniejszy detektor
python src/detect_faces.py --force            # przelicz od nowa

# 2. Wizualizacja 3D (domyslnie: UMAP, twarz > swiatlo, kolor = grupa)
python src/visualize_3d.py                          # UMAP + rozpychanie grup
python src/visualize_3d.py --explode 3 --spread 15  # wieksze odstepy miedzy grupami
python src/visualize_3d.py --eps 0.25               # wiecej, drobniejszych grup
python src/visualize_3d.py --reduce TSNE --perplexity 5
python src/visualize_3d.py --light-strength 0       # surowe embeddingi do porownania

# 3. (opcjonalnie) etykiety grup do kolorowania / LDA
python src/group_faces.py
python src/visualize_3d.py --reduce LDA --labels emb/labels_auto.csv --color-by label
```

## Uczenie metryki z etykiet (najlepsze rozdzielanie osob)

Jesli oznaczysz kto-jest-kim, mozesz NAUCZYC przeksztalcenia, ktore sciaga ta
sama osobe razem i odpycha rozne osoby - naprawia to przypadki, gdy ta sama
osoba ladowala daleko, a dwie rozne blisko.

Etykiety nadajesz ukladajac wyciete twarze w podfoldery:

```
emb_images/
  Ania/   <base>#twarz#0.jpg ...
  Marek/  <base>#twarz#1.jpg ...
```

Potem:

```bash
# Pomiar PRZED/PO (recall@1 = czy najblizszy sasiad to ta sama osoba; wyzej=lepiej)
python src/evaluate.py

# Wizualizacja z nauczona metryka (lmnn zwykle najlepszy)
python src/visualize_3d.py --metric lmnn --labels-from-folders --explode 3
python src/visualize_3d.py --metric lda  --labels-from-folders
python src/visualize_3d.py --metric nca  --labels-from-folders
```

Metody: `lda` (scikit-learn), `nca` (scikit-learn), `lmnn` (pip install metric-learn).
`evaluate.py` pokaze, ktora najlepiej dziala na Twoich danych.

## Mniej falszywych twarzy (szumy, zwierzeta)

Detektor podaje pewnosc detekcji - szumy i pyski zwierzat dostaja niski wynik,
wiec wystarczy odfiltrowac je progiem:

- `--min-score N` (ArcFace) - odrzuca detekcje ponizej pewnosci N. Domyslnie 0.6;
  podnies do 0.7-0.8, jesli wciaz lapie zwierzeta/szum.
- `--min-detect-size N` - odrzuca male ramki twarzy (domyslnie 40 px).
- `--dlib-model cnn` - dokladniejszy detektor dla backendu dlib.
- pewnosc (`score`) jest zapisywana w `emb/meta.csv`.

### Psy i grafiki lapane jako twarze

Detektor jest trenowany na ludzkich twarzach, ale czasem z wysoka pewnoscia
lapie pyski zwierzat albo twarze z rysunkow/grafik - wtedy sam `--min-score`
nie pomaga. Wlacz semantyczny filtr CLIP, ktory klasyfikuje kazdy kadr
("ludzka twarz" vs "pies/zwierze" vs "rysunek/grafika") i odrzuca nie-ludzkie:

```bash
pip install open_clip_torch torch          # najlepiej wersja CUDA
python src/detect_faces.py --device gpu --min-score 0.7 --clip-filter
```

Decyzja jest WZGLEDNA (twarz musi tylko wygrac z "pies/grafika", a nie przebic
sztywny prog), wiec twarze pod katem / z profilu / slabej jakosci przechodza.
Czulosc:
- `--clip-threshold` (domyslnie 0.15) - dolny prog anty-smieci, obniz jesli
  wycina prawdziwe twarze,
- `--clip-margin` (domyslnie 0.0) - wymagana przewaga "twarzy"; ustaw UJEMNY
  (np. -0.03), zeby bylo lagodniej dla trudnych twarzy.

Strojenie jako STRONA (zalecane) - CLIP liczy raz, a Ty suwakami na zywo
widzisz, ktore kadry zostaja, a ktore wypadaja:

```bash
python src/clip_export.py --source images --device gpu   # tworzy clip_tuning.html
```

`--source images` uruchamia detektor na Twoich zdjeciach i pokazuje WSZYSTKIE
kandydatury - prawdziwe twarze ORAZ falszywe (psy, grafiki) - czyli dokladnie to,
co filtr ma rozdzielic. (`--source crops` uzywa gotowych twarzy z emb_images.)
Otworz `clip_tuning.html`, przesuwaj prog i margines (galeria "Wypada" sortowana
od granicznych - tam wypatrz prawdziwe twarze), a strona pokaze gotowa komende
detect_faces z Twoimi wartosciami.

Wersja tekstowa (bez przegladarki):

```bash
python src/clip_tune.py --dir emb_images --clip-threshold 0.14 --clip-margin -0.02
```

### Auto-kalibracja BEZ etykiet (gdy nic nie posortowane)

Nie masz przykladow "twarz"/"nie-twarz"? CLIP-owa przewaga (human-reject) sama
uklada sie w dwa skupiska, a granice znajdzie automatycznie Otsu / mieszanka
gaussowska:

```bash
python src/clip_auto.py --source images --device gpu
```

Wypisze automatyczny `--clip-margin`, oceni, czy skupiska sa wyraznie rozdzielone,
i zbuduje `clip_tuning.html` juz ustawiony na tej granicy (mozesz dostroic suwakiem).
Potem uzywasz podanej komendy detect_faces.

### Test filtra (auto-strojenie liczbami, z etykietami)

Najszybszy sposob na poprawe: dajesz przyklady "twarz" i "nie-twarz", a skrypt
sam znajduje najlepszy prog/margines (F1) i pokazuje, co jeszcze MYLI.

```bash
# TWARZE = foldery osob w emb_images, NIE-TWARZE = nierozpoznane/zwierzeta:
python src/clip_eval.py --auto --device gpu --html

# albo wlasne foldery z przykladami:
python src/clip_eval.py --faces-dir eval/face --not-faces-dir eval/junk --html
```

Wypisze gotowa komende z najlepszymi `--clip-threshold/--clip-margin`, macierz
pomylek oraz (z `--html`) `clip_eval.html` z dwiema galeriami: SMIECI blednie
zostawione (dla nich dodaj opisy w `face_filter.py`) i TWARZE blednie odrzucone.
Iteracja: zobacz pomylki -> dodaj/popraw opisy w `face_filter.py` -> uruchom test
ponownie i porownaj F1.

Czyszczenie tego, co juz sie wykrylo (na podstawie zapisanej pewnosci):

```bash
python src/prune.py --min-score 0.7           # podglad: co zostanie usuniete
python src/prune.py --min-score 0.7 --apply   # faktyczne usuniecie
```

## Zaszumiony zbior (zwierzeta, smieci, male kadry, pomylki)

Pipeline jest odporny na realne dane:

- **Foldery-nie-osoby** (`nierozpoznane`, `zwierzeta`, `inne`, `unknown`...) sa
  automatycznie pomijane - nie trafiaja do uczenia metryki ani do oceny.
  Liste zmienisz przez `--ignore-folders nazwa1 nazwa2`.
- **Male kadry** (np. 32x32) daja bezuzyteczne embeddingi - odrzuca je
  `--min-face-size` (domyslnie 50 px; `0` wylacza filtr).
- **Pomylki w etykietach** - `labels_audit.py` znajduje twarze blizsze innej
  osobie niz wlasnej i zapisuje raport `emb/suspected_mislabels.csv` do recznego
  sprawdzenia. Flaga `--drop-suspected` pomija je przy uczeniu metryki.

```bash
python src/labels_audit.py                                   # raport pomylek
python src/evaluate.py --min-face-size 64 --drop-suspected   # uczciwszy pomiar
python src/visualize_3d.py --metric lmnn --labels-from-folders --drop-suspected --explode 3
```

## Interaktywna wizualizacja

Zdjecia tej samej osoby ukladaja sie na SIATCE (nie nakladaja sie), a w oknie:

- **kliknij dowolne zdjecie** -> kamera przybliza do calej grupy tej osoby
  (u gory pojawia sie nazwa osoby i liczba zdjec),
- **klawisz R** -> reset widoku.

Sterowanie ukladem:

```bash
python src/visualize_3d.py --metric lmnn --labels-from-folders --only-labeled
python src/visualize_3d.py --metric lmnn --labels-from-folders --cell 0.8 --explode 3
```

- `--cell`    odstep zdjec w obrebie grupy (wieksze = luzniejsza siatka),
- `--explode` odstep miedzy grupami, `--spread` globalna skala,
- `--no-interactive` wylacza klikanie.

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
