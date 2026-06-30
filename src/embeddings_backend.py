"""
Backendy embeddingow twarzy ukryte za wspolnym interfejsem.

- ArcFaceBackend : insightface / ArcFace (512-D). Domyslny. Dziala na GPU (CUDA)
                   przez onnxruntime-gpu. Samo dociaga biblioteki CUDA z pip-owych
                   wheeli nvidia-*-cu12 (preload przez ctypes), zeby onnxruntime
                   znalazl np. cublasLt64_12.dll. Sprawdza, czy CUDA NAPRAWDE wstala.
- DlibBackend    : face_recognition (dlib, 128-D). Lzejszy.

Filtry redukujace falszywe twarze: min_score (ArcFace), min_size (bok ramki w px).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from preprocess import crop_face, clahe_normalize, brightness


@dataclass
class Face:
    bbox: tuple
    embedding: np.ndarray
    crop: np.ndarray
    brightness: float
    score: float = 1.0


def _box_min_side(loc) -> int:
    top, right, bottom, left = loc
    return min(bottom - top, right - left)


def pick_providers(device: str, available):
    """Wybiera providery ONNX Runtime. Zwraca (providers, ctx_id, etykieta)."""
    has_cuda = "CUDAExecutionProvider" in available
    if device in ("auto", "gpu") and has_cuda:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0, "GPU (CUDA)"
    return ["CPUExecutionProvider"], -1, "CPU"


def nvidia_bin_dirs():
    """
    Foldery z bibliotekami CUDA z pip-owych wheeli nvidia-*-cu12.

    'nvidia' to namespace package (PEP 420), wiec nvidia.__file__ bywa None -
    katalogi bierzemy z nvidia.__path__. Na Windows DLL-e leza w <lib>/bin.
    """
    import os
    import glob
    try:
        import nvidia
    except Exception:  # noqa: BLE001
        return []
    roots = list(getattr(nvidia, "__path__", []) or [])
    if not roots and getattr(nvidia, "__file__", None):
        roots = [os.path.dirname(nvidia.__file__)]
    dirs = []
    for root in roots:
        for sub in ("bin", "lib"):
            dirs += [d for d in glob.glob(os.path.join(root, "*", sub)) if os.path.isdir(d)]
    return sorted(set(dirs))


def register_nvidia_dll_dirs(verbose: bool = False):
    """
    Udostepnia onnxruntime biblioteki CUDA z pip-owych wheeli nvidia-*-cu12:
      1) os.add_dll_directory dla kazdego folderu,
      2) dorzuca foldery na poczatek PATH,
      3) WSTEPNIE laduje wszystkie .dll przez ctypes (kilka przebiegow, bo maja
         wzajemne zaleznosci) - cublasLt64_12.dll itp. trafia do pamieci, zanim
         onnxruntime zaladuje swoj provider CUDA.
    """
    import os
    import glob

    bin_dirs = nvidia_bin_dirs()
    if verbose:
        print(f"Foldery CUDA znalezione: {len(bin_dirs)}")
        for d in bin_dirs:
            print("   ", d)
    if os.name != "nt" or not bin_dirs:
        return bin_dirs

    for d in bin_dirs:
        try:
            os.add_dll_directory(d)
        except OSError:
            pass
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")

    import ctypes
    dlls = []
    for d in bin_dirs:
        dlls += glob.glob(os.path.join(d, "*.dll"))
    pending, loaded = list(dlls), 0
    for _ in range(5):  # kilka przebiegow rozwiazuje wzajemne zaleznosci
        still = []
        for dll in pending:
            try:
                ctypes.WinDLL(dll)
                loaded += 1
            except OSError:
                still.append(dll)
        if len(still) == len(pending):
            break
        pending = still
    if verbose:
        print(f"Zaladowano {loaded}/{len(dlls)} DLL-i CUDA (nieudane: {len(pending)})")
    return bin_dirs


def _real_providers(app) -> set:
    """Faktyczne providery uzyte przez modele insightface (po zaladowaniu sesji)."""
    real = set()
    for m in (getattr(app, "models", {}) or {}).values():
        sess = getattr(m, "session", None)
        if sess is not None:
            try:
                real.update(sess.get_providers())
            except Exception:  # noqa: BLE001
                pass
    return real


class DlibBackend:
    name = "dlib"

    def __init__(self, normalize_light: bool = True, model: str = "hog", min_size: int = 0):
        self.normalize_light = normalize_light
        self.model = model
        self.min_size = min_size

    def get_faces(self, image_rgb: np.ndarray):
        import face_recognition

        locations = face_recognition.face_locations(image_rgb, model=self.model)
        faces, rejected = [], 0
        for loc in locations:
            if self.min_size and _box_min_side(loc) < self.min_size:
                rejected += 1
                continue
            crop = crop_face(image_rgb, loc)
            if crop.size == 0:
                continue
            bright = brightness(crop)
            enc_input = clahe_normalize(crop) if self.normalize_light else crop
            h, w = enc_input.shape[:2]
            encs = face_recognition.face_encodings(enc_input, [(0, w, h, 0)], num_jitters=1)
            if not encs:
                continue
            faces.append(Face(bbox=loc, embedding=encs[0], crop=crop, brightness=bright, score=1.0))
        self.last_rejected = rejected
        return faces


CUDA_FIX_HINT = (
    "GPU zazadane, ale CUDA NIE wstala - liczy na CPU (wolno).\n"
    "    Sprawdz diagnostyka:  python src/gpu_check.py\n"
    "    Najczestsze przyczyny:\n"
    "      - brak bibliotek CUDA 12: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 "
    "nvidia-cuda-runtime-cu12 nvidia-cufft-cu12\n"
    "      - niezgodna wersja: onnxruntime-gpu musi pasowac do CUDA 12 (pip show onnxruntime-gpu)\n"
    "      - brak sterownika NVIDIA / karta nie-NVIDIA."
)


class ArcFaceBackend:
    name = "arcface"

    def __init__(self, normalize_light: bool = False, det_size: int = 640,
                 min_score: float = 0.6, min_size: int = 0, device: str = "auto"):
        register_nvidia_dll_dirs()  # zanim onnxruntime zacznie ladowac CUDA
        try:
            from insightface.app import FaceAnalysis
            import onnxruntime as ort
        except ImportError as exc:  # noqa: BLE001
            raise ImportError("Backend 'arcface' wymaga: pip install insightface onnxruntime-gpu") from exc

        if hasattr(ort, "preload_dlls"):   # onnxruntime 1.21+: dociaga CUDA z pip wheeli
            try:
                ort.preload_dlls()
            except Exception:  # noqa: BLE001
                pass

        self.min_score = min_score
        self.min_size = min_size
        providers, ctx_id, _ = pick_providers(device, ort.get_available_providers())

        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

        gpu_active = "CUDAExecutionProvider" in _real_providers(self.app)
        self.device_used = "GPU (CUDA)" if gpu_active else "CPU"
        if device in ("gpu", "auto") and not gpu_active and "CUDAExecutionProvider" in providers:
            print("UWAGA: " + CUDA_FIX_HINT)
        print(f"ArcFace na: {self.device_used} | det_size={det_size}")

    def get_faces(self, image_rgb: np.ndarray):
        results = self.app.get(image_rgb[:, :, ::-1])  # BGR
        faces, rejected = [], 0
        for r in results:
            score = float(getattr(r, "det_score", 1.0))
            x1, y1, x2, y2 = (int(v) for v in r.bbox)
            loc = (y1, x2, y2, x1)
            if score < self.min_score or (self.min_size and _box_min_side(loc) < self.min_size):
                rejected += 1
                continue
            crop = crop_face(image_rgb, loc)
            emb = getattr(r, "normed_embedding", None)
            if emb is None:
                emb = r.embedding
            faces.append(Face(bbox=loc, embedding=np.asarray(emb, dtype=np.float32),
                              crop=crop, brightness=brightness(crop), score=score))
        self.last_rejected = rejected
        return faces


def get_backend(name: str, normalize_light: bool = True, min_score: float = 0.6,
                min_size: int = 0, dlib_model: str = "hog", device: str = "auto",
                det_size: int = 640):
    name = name.lower()
    if name == "dlib":
        return DlibBackend(normalize_light=normalize_light, model=dlib_model, min_size=min_size)
    if name == "arcface":
        return ArcFaceBackend(normalize_light=normalize_light, det_size=det_size,
                              min_score=min_score, min_size=min_size, device=device)
    raise ValueError(f"Nieznany backend: {name} (dostepne: dlib, arcface)")
