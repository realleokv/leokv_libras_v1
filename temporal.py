"""Validação temporal compartilhada entre coleta, treinamento e inferência."""

import numpy as np
from config import TOTAL_FEATURES_DIM, SEQ_LENGTH, MAX_FRAME_GAP, SEQUENCE_SECONDS


def validate_frames(features, timestamps, minimum=2):
    x, t = np.asarray(features, np.float32), np.asarray(timestamps, np.float64)
    if x.ndim != 2 or x.shape[1] != TOTAL_FEATURES_DIM or not minimum <= len(x) <= 200:
        raise ValueError("Gravação com quantidade ou dimensão de frames inválida.")
    if t.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(t).all():
        raise ValueError("Gravação contém valores inválidos.")
    gaps = np.diff(t)
    if np.any(gaps <= 0) or np.any(gaps > MAX_FRAME_GAP):
        raise ValueError(
            "Gravação interrompida. Repita mantendo a mão visível e a conexão estável."
        )
    return x, t


def resample_sequence(features, timestamps):
    x, t = validate_frames(features, timestamps, minimum=6)
    if not SEQUENCE_SECONDS - 0.25 <= t[-1] - t[0] <= SEQUENCE_SECONDS + 0.3:
        raise ValueError(
            f"Grave aproximadamente {SEQUENCE_SECONDS:g} segundos contínuos por gesto."
        )
    target = np.linspace(t[0], t[-1], SEQ_LENGTH)
    out = np.column_stack(
        [np.interp(target, t, x[:, i]) for i in range(TOTAL_FEATURES_DIM)]
    )
    # Flags de presença nunca são inventadas por interpolação.
    nearest = np.abs(t[:, None] - target).argmin(axis=0)
    out[:, [0, 73, 155]] = x[nearest][:, [0, 73, 155]]
    return out.astype(np.float32)
