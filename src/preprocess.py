"""
Wstepne przetwarzanie kadrow twarzy.

- crop_face       : bezpieczne wyciecie twarzy z obrazu wg lokalizacji
- brightness      : srednia luminancja kadru -> kowiata do "regresji swiatla"
- clahe_normalize : wyrownanie oswietlenia (CLAHE na kanale L w przestrzeni LAB)

CLAHE wyrownuje lokalny kontrast/oswietlenie, dzieki czemu model patrzy
bardziej na rysy twarzy niz na cienie. Wymaga OpenCV (opencv-python).
"""
from __future__ import annotations

import numpy as np


def crop_face(image_rgb: np.ndarray, location: tuple) -> np.ndarray:
    """location = (top, right, bottom, left) w konwencji face_recognition."""
    top, right, bottom, left = location
    top, left = max(0, top), max(0, left)
    bottom = min(image_rgb.shape[0], bottom)
    right = min(image_rgb.shape[1], right)
    return image_rgb[top:bottom, left:right]


def brightness(crop_rgb: np.ndarray) -> float:
    """Srednia luminancja wg ITU-R BT.601 (0..255)."""
    if crop_rgb.size == 0:
        return 0.0
    r = crop_rgb[..., 0].astype(np.float32)
    g = crop_rgb[..., 1].astype(np.float32)
    b = crop_rgb[..., 2].astype(np.float32)
    return float((0.299 * r + 0.587 * g + 0.114 * b).mean())


def clahe_normalize(crop_rgb: np.ndarray) -> np.ndarray:
    """Wyrownanie oswietlenia metoda CLAHE. Zwraca obraz RGB uint8."""
    import cv2

    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
