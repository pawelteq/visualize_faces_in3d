"""
Semantyczny filtr twarzy (opcjonalny) - zostawia tylko PRAWDZIWE, ludzkie twarze
z fotografii. Odrzuca: zwierzeta, rysunki/anime, rendery 3D / CGI, postacie z gier,
rzezby/figurki, maski/kaski/roboty, loga i grafiki.

Decyzja jest WZGLEDNA: kadr przechodzi, gdy najlepszy opis "prawdziwej ludzkiej
twarzy" wygrywa z najlepszym opisem "nie-twarzy" o zadany margines. Twarze pod
katem / z profilu wciaz przypominaja czlowieka bardziej niz psa czy render, wiec
zostaja - ale realistyczny render czy kask dostaja teraz wysoki "reject".

Wymaga: pip install open_clip_torch torch  (najlepiej wersja CUDA).
Uzywany tylko gdy detect_faces.py / clip_export.py dostanie --clip-filter.
"""
from __future__ import annotations

import numpy as np

HUMAN_PROMPTS = [
    "a real photograph of a human face",
    "a candid photo of a real person's face",
    "a close-up photo of a person's face",
    "a side profile of a real human face",
    "a real human face seen from an angle",
    "a tilted human face in a photo",
    "a blurry photo of a real human face",
    "a low quality photo of a person's face",
    "a partially visible human face",
    "a real person wearing glasses",
    "a selfie of a real person",
]
REJECT_PROMPTS = [
    "a photo of a dog",
    "a photo of a cat",
    "a photo of an animal",
    "an animal's face",
    "a drawing or cartoon",
    "an anime character",
    "a digital illustration",
    "a painting of a face",
    "a 3d render of a face",
    "a cgi character",
    "a computer generated face",
    "a video game character",
    "a screenshot from a video game",
    "a statue or sculpture of a face",
    "a clay figurine or doll",
    "a mask or helmet",
    "a robot or cyborg",
    "a logo or text",
    "a screenshot of an app",
    "an object that is not a face",
]


def decide_human(sims, labels, threshold: float, margin: float):
    """
    sims/labels : podobienstwa crop<->prompty oraz ich kategorie ("human"/"reject").
    Przechodzi, gdy:  human_max - reject_max >= margin  ORAZ  human_max >= threshold.
    """
    sims = np.asarray(sims, dtype=float)
    human = max((s for s, l in zip(sims, labels) if l == "human"), default=-1.0)
    reject = max((s for s, l in zip(sims, labels) if l == "reject"), default=-1.0)
    is_human = (human >= threshold) and (human - reject >= margin)
    return bool(is_human), float(human), float(reject)


class ClipFaceFilter:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k",
                 device: str = "auto", threshold: float = 0.15, margin: float = 0.0):
        try:
            import open_clip
            import torch
        except ImportError as exc:  # noqa: BLE001
            raise SystemExit("Filtr CLIP wymaga: pip install open_clip_torch torch") from exc

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.threshold = threshold
        self.margin = margin

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device)
        self.model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)

        prompts = HUMAN_PROMPTS + REJECT_PROMPTS
        self.labels = ["human"] * len(HUMAN_PROMPTS) + ["reject"] * len(REJECT_PROMPTS)
        tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            tf = self.model.encode_text(tokens)
            tf /= tf.norm(dim=-1, keepdim=True)
        self.text_features = tf
        print(f"Filtr CLIP: {model_name}/{pretrained} na {device} "
              f"(prog={threshold}, margines={margin}, opisow={len(prompts)})")

    def scores(self, crop_rgb: np.ndarray):
        """Zwraca (is_human, human_max, reject_max) dla kadru."""
        from PIL import Image
        img = self.preprocess(Image.fromarray(crop_rgb)).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            f = self.model.encode_image(img)
            f /= f.norm(dim=-1, keepdim=True)
            sims = (f @ self.text_features.T).squeeze(0).cpu().numpy()
        return decide_human(sims, self.labels, self.threshold, self.margin)

    def is_human_face(self, crop_rgb: np.ndarray):
        return self.scores(crop_rgb)
