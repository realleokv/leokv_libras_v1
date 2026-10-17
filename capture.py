"""Coleta no servidor com proveniência e validação de continuidade."""

import uuid
import numpy as np
from config import BACKGROUND_LABEL, MAX_FRAME_GAP, SEQUENCE_SECONDS
from database import text_field
from temporal import validate_frames, resample_sequence


class Capture:
    def __init__(self, sign, data, now):
        self.id = uuid.uuid4().hex
        self.sign = sign
        self.kind = data.get("kind")
        if self.kind not in ("static", "dynamic"):
            raise ValueError("Escolha o modo de coleta.")
        if sign["label"] != BACKGROUND_LABEL and (self.kind == "dynamic") != bool(
            sign["is_dynamic"]
        ):
            raise ValueError("Modo de coleta não corresponde ao sinal.")
        self.signer = text_field(data.get("signer"), "Código da pessoa", 40)
        self.session = text_field(data.get("session"), "Código da sessão", 40)
        self.target = data.get("count", 12)
        if type(self.target) is not int or not 6 <= self.target <= 30:
            raise ValueError("Use entre 6 e 30 frames estáticos por gravação.")
        self.started = now
        self.frames, self.timestamps = [], []
        self.signature = None
        self.last_frame_time = None

    def add(self, extracted, timestamp, now):
        if now - self.started > 15:
            raise ValueError("Tempo de coleta esgotado. Tente novamente.")
        if self.sign["label"] != BACKGROUND_LABEL:
            if extracted["hands_detected"] < (2 if self.sign["two_hands"] else 1):
                raise ValueError(
                    "Coleta cancelada: uma mão necessária saiu do enquadramento."
                )
            if self.sign["requires_body"] and not extracted["body_detected"]:
                raise ValueError("Coleta cancelada: mostre rosto e ombros.")
            signature = extracted.get("tracking_signature")
            if self.signature is not None and signature != self.signature:
                raise ValueError("Coleta cancelada: a lateralidade das mãos mudou.")
            self.signature = signature
        if (
            self.last_frame_time is not None
            and not 0 < timestamp - self.last_frame_time <= MAX_FRAME_GAP
        ):
            raise ValueError("Coleta interrompida pela conexão. Repita o gesto.")
        self.last_frame_time = timestamp
        # Espaça as amostras estáticas em vez de contar todos os frames semelhantes.
        if (
            self.kind == "static"
            and self.timestamps
            and timestamp - self.timestamps[-1] < 0.20
        ):
            return False
        self.frames.append(extracted["features"].copy())
        self.timestamps.append(timestamp)
        if self.kind == "static":
            done = len(self.frames) >= self.target
        else:
            done = (
                len(self.frames) > 1
                and timestamp - self.timestamps[0] >= SEQUENCE_SECONDS
            )
        if done:
            validate_frames(self.frames, self.timestamps)
            if self.kind == "dynamic":
                resample_sequence(self.frames, self.timestamps)
        return done

    def progress(self):
        current = (
            len(self.frames)
            if self.kind == "static"
            else (
                self.timestamps[-1] - self.timestamps[0]
                if len(self.timestamps) > 1
                else 0
            )
        )
        target = self.target if self.kind == "static" else SEQUENCE_SECONDS
        return {
            "active": True,
            "current": round(current, 1),
            "target": target,
            "label": self.sign["label"],
        }

    def save(self, database):
        database.save_recording(
            self.id,
            self.sign["label"],
            self.kind,
            self.signer,
            self.session,
            np.array(self.frames, np.float32),
            np.array(self.timestamps, np.float64),
        )
