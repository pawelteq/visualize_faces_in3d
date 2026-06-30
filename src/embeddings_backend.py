"""
Backendy embeddingow twarzy ukryte za wspolnym interfejsem.

- DlibBackend    : face_recognition (dlib ResNet, 128-D). Lzejszy.
- ArcFaceBackend : insightface / ArcFace (512-D) - duzo wieksza odpornosc
                   na swiatlo i poze. Domyslny. Wymaga: pip install insightface onnxruntime
                   (przy pierwszym uruchomieniu pobiera model).

Kazdy backend zwraca liste obiektow Face (bbox, embedding, podglad, jasnosc).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from preprocess import crop_face, clahe_normalize, brightness


@dataclass
class Face:
    bbox: tuple          # (top, right, bottom, left)
    embedding: np.ndarray
    crop: np.ndarray     # RGB uint8 - oryginalny kadr (do podgladu)
    brightness: float    # jasnosc oryginalnego kadru (kowiata)


class DlibBackend:
    name = "dlib"

    def __init__(self, normalize_light: bool = True):
        self.normalize_light = normalize_light

    def get_faces(self, image_rgb: np.ndarray) -> list[Face]:
        import face_recognition

        locations = face_recognition.face_locations(image_rgb)
        faces: list[Face] = []
        for loc in locations:
            crop = crop_face(image_rgb, loc)
            if crop.size == 0:
                continue
            bright = brightness(crop)
            enc_input = clahe_normalize(crop) if self.normalize_light else crop
            h, w = enc_input.shape[:2]
            # kodujemy z znormalizowanego kadru (caly kadr = jedna twarz)
            encs = face_recognition.face_encodings(enc_input, [(0, w, h, 0)], num_jitters=1)
            if not encs:
                continue
            faces.append(Face(bbox=loc, embedding=encs[0], crop=crop, brightness=bright))
        return faces


class ArcFaceBackend:
    name = "arcface"

    def __init__(self, normalize_light: bool = False, det_size: int = 640):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "Backend 'arcface' wymaga: pip install insightface onnxruntime"
            ) from exc
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size))

    def get_faces(self, image_rgb: np.ndarray) -> list[Face]:
        # insightface oczekuje BGR; alignment i normalizacja sa wewnatrz modelu
        results = self.app.get(image_rgb[:, :, ::-1])
        faces: list[Face] = []
        for r in results:
            x1, y1, x2, y2 = (int(v) for v in r.bbox)
            loc = (y1, x2, y2, x1)  # top, right, bottom, left
            crop = crop_face(image_rgb, loc)
            emb = getattr(r, "normed_embedding", None)
            if emb is None:
                emb = r.embedding
            faces.append(
                Face(
                    bbox=loc,
                    embedding=np.asarray(emb, dtype=np.float32),
                    crop=crop,
                    brightness=brightness(crop),
                )
            )
        return faces


def get_backend(name: str, normalize_light: bool = True):
    name = name.lower()
    if name == "dlib":
        return DlibBackend(normalize_light=normalize_light)
    if name == "arcface":
        return ArcFaceBackend(normalize_light=normalize_light)
    raise ValueError(f"Nieznany backend: {name} (dostepne: dlib, arcface)")
