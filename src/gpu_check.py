"""
Diagnostyka GPU dla onnxruntime / ArcFace.

Pokazuje wersje onnxruntime, dostepne providery, znalezione biblioteki CUDA
(z pip-owych wheeli nvidia-*) oraz sprawdza, czy kluczowe DLL-e CUDA daja sie
zaladowac po nazwie. Jesli wszystkie sa OK, a detekcja dalej liczy na CPU -
najpewniej wersja onnxruntime-gpu nie pasuje do zainstalowanej CUDA.

Uzycie:
    python src/gpu_check.py
"""
from __future__ import annotations

import os
import ctypes

from embeddings_backend import register_nvidia_dll_dirs, nvidia_bin_dirs


REQUIRED_DLLS = [
    "cublasLt64_12.dll", "cublas64_12.dll", "cudnn64_9.dll",
    "cufft64_11.dll", "cudart64_12.dll", "nvrtc64_120_0.dll",
]


def main() -> None:
    dirs = register_nvidia_dll_dirs(verbose=True)
    print("\nFoldery CUDA (pip nvidia-*):")
    for d in (dirs or nvidia_bin_dirs()):
        print("  ", d)
    if not dirs:
        print("  (brak - zainstaluj: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 "
              "nvidia-cuda-runtime-cu12 nvidia-cufft-cu12)")

    try:
        import onnxruntime as ort
        print(f"\nonnxruntime: {ort.__version__}")
        print("dostepne providery:", ort.get_available_providers())
    except Exception as exc:  # noqa: BLE001
        print("\nNie udalo sie zaimportowac onnxruntime:", exc)

    print("\nLadowanie kluczowych DLL-i CUDA:")
    if os.name != "nt":
        print("  (test dotyczy Windows)")
        return
    all_ok = True
    for name in REQUIRED_DLLS:
        try:
            ctypes.WinDLL(name)
            print(f"  OK   {name}")
        except OSError:
            all_ok = False
            print(f"  BRAK {name}")

    print()
    if all_ok:
        print("Wszystkie biblioteki OK. Jesli ArcFace dalej liczy na CPU, to znaczy,")
        print("ze wersja onnxruntime-gpu nie pasuje do CUDA 12 - sprawdz `pip show onnxruntime-gpu`.")
    else:
        print("Brakuje bibliotek powyzej - doinstaluj pasujace pakiety nvidia-*-cu12.")


if __name__ == "__main__":
    main()
