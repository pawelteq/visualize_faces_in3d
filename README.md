# Face Embeddings 3D

Wykrywa twarze na zdjeciach, generuje embeddingi (wektory cech) i wizualizuje
je w przestrzeni 3D. Twarze podobne do siebie laduja blisko siebie.

## Pipeline

```
images/        zdjecia wejsciowe
   |  detect_faces.py   (face_recognition / dlib)
   v
emb/           embeddingi 128-D  (<nazwa>#<i>.npy)
emb_images/    wyciete twarze    (<nazwa>#twarz#<i>.jpg)
   |  visualize_3d.py   (PCA / t-SNE -> PyVista)
   v
okno 3D z miniaturami twarzy
```

## Instalacja

```bash
pip install -r requirements.txt
```

Uwaga: `face_recognition` wymaga `dlib`, ktory na Windows potrzebuje
**CMake** oraz **Visual Studio Build Tools (C++)**. Jesli chcesz tylko
obejrzec gotowe embeddingi (folder `emb/` juz je zawiera), wystarczy:

```bash
pip install numpy Pillow scikit-learn pyvista
```

## Uzycie

Detekcja (zapisuje embeddingi i wyciete twarze):

```bash
python src/detect_faces.py            # przetwarza nowe zdjecia
python src/detect_faces.py --force    # przelicza wszystko od nowa
```

Wizualizacja 3D:

```bash
python src/visualize_3d.py               # PCA (domyslnie)
python src/visualize_3d.py --method TSNE # t-SNE
```

Skrypty same znajduja foldery `images/`, `emb/`, `emb_images/` niezaleznie
od tego, czy uruchamiasz je z korzenia projektu, czy z `src/`.

## Struktura projektu

```
programik/
  src/
    detect_faces.py     # detekcja twarzy -> embeddingi
    visualize_3d.py     # wizualizacja 3D embeddingow
  images/               # zdjecia wejsciowe
  emb/                  # embeddingi .npy
  emb_images/           # wyciete twarze .jpg
  requirements.txt
  README.md
```

## Co sie zmienilo wzgledem starej wersji

- Usunieto `wizualizacja.py` (matplotlib) - mial bledne dopasowanie nazw
  plikow; `visualize_3d.py` (PyVista) to lepsze rozwiazanie.
- Z detekcji usunieto nieuzywany `torch`/`torchvision` (detekcja i tak
  liczona jest przez dlib na CPU).
- Nazwy embeddingow sa deterministyczne (bez timestampu), wiec ponowne
  uruchomienie nie tworzy duplikatow.
- Usunieto folder `sorter/` - to byl osobny, niepowiazany projekt
  (sorter plikow oparty na Ollama/PyQt5), nie nalezal do tego pipeline'u.

> Pierwsze uruchomienie: odpal `cleanup.ps1`, aby przeniesc skrypty do `src/`
> i usunac przestarzale pliki. Skrypty dzialaja takze bez tego kroku.
