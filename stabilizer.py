"""Decisões temporais por conexão e emissão por execução do gesto."""

from collections import deque
import numpy as np
from config import (
    STATIC_THRESHOLD,
    DYNAMIC_THRESHOLD,
    MIN_MARGIN,
    BACKGROUND_LABEL,
    MOTION_VELOCITY_THRESHOLD,
    MAX_FRAME_GAP,
    SEQUENCE_SECONDS,
)
from temporal import resample_sequence


def result(
    status, label="Sinal não reconhecido", score=0.0, dynamic=False, emitted=None
):
    return {
        "status": status,
        "display_label": label,
        "score": round(score * 100, 1),
        "is_dynamic": dynamic,
        "emitted": emitted,
    }


class TemporalStabilizer:
    def __init__(self):
        self.thresholds = {"static": STATIC_THRESHOLD, "dynamic": DYNAMIC_THRESHOLD}
        self.reset()

    def reset(self):
        self.frames = deque(maxlen=80)
        self.static_votes = deque()
        self.last_timestamp = None
        self.signature = None
        self.emitted_label = None
        self.neutral_since = None
        self.last_dynamic_check = -100.0
        self.dynamic_candidate = None
        self.dynamic_votes = 0

    def _neutral(self, timestamp):
        if self.neutral_since is None:
            self.neutral_since = timestamp
        if timestamp - self.neutral_since >= 0.4:
            self.emitted_label = None

    def process_step(self, extracted, timestamp, engine, signs):
        x, velocity = extracted["features"], extracted["velocity"]
        signature = extracted.get("tracking_signature")
        if (
            self.last_timestamp is not None
            and timestamp - self.last_timestamp > MAX_FRAME_GAP
        ):
            self.reset()
        self.last_timestamp = timestamp
        if not extracted["hands_detected"]:
            self.frames.clear()
            self.static_votes.clear()
            self.dynamic_candidate, self.dynamic_votes = None, 0
            self._neutral(timestamp)
            return result("no_hands", "Aguardando mãos")
        if self.signature != signature:
            self.frames.clear()
            self.static_votes.clear()
            self.dynamic_candidate, self.dynamic_votes = None, 0
        self.signature = signature
        if not engine.available("static") and not engine.available("dynamic"):
            return result("untrained", "Faça a coleta e treine um modelo v2")
        self.frames.append((timestamp, x.copy(), velocity))
        while (
            len(self.frames) > 1
            and timestamp - self.frames[0][0] > SEQUENCE_SECONDS + 0.2
        ):
            self.frames.popleft()
        if velocity >= MOTION_VELOCITY_THRESHOLD:
            self.static_votes.clear()
        candidate, score, dynamic = None, 0.0, False
        # Movimento em qualquer parte da janela, inclusive se o gesto terminou parado.
        if (
            engine.available("dynamic")
            and len(self.frames) >= 6
            and timestamp - self.frames[0][0] >= SEQUENCE_SECONDS - 0.25
            and any(f[2] >= MOTION_VELOCITY_THRESHOLD for f in self.frames)
            and timestamp - self.last_dynamic_check >= 0.15
        ):
            self.last_dynamic_check = timestamp
            try:
                sequence = resample_sequence(
                    [f[1] for f in self.frames], [f[0] for f in self.frames]
                )
                label, score, margin = engine.predict("dynamic", sequence)
                if label == BACKGROUND_LABEL:
                    self._neutral(timestamp)
                if (
                    label
                    and label != BACKGROUND_LABEL
                    and score >= self.thresholds["dynamic"]
                    and margin >= MIN_MARGIN
                ):
                    self.dynamic_votes = (
                        self.dynamic_votes + 1 if label == self.dynamic_candidate else 1
                    )
                    self.dynamic_candidate = label
                    if self.dynamic_votes >= 2:
                        candidate, dynamic = label, True
                else:
                    self.dynamic_candidate, self.dynamic_votes = None, 0
            except ValueError:
                self.dynamic_candidate, self.dynamic_votes = None, 0
        if (
            candidate is None
            and engine.available("static")
            and velocity < MOTION_VELOCITY_THRESHOLD
        ):
            label, static_score, margin = engine.predict("static", x)
            score = max(score, static_score)
            if label == BACKGROUND_LABEL:
                self.static_votes.clear()
                self._neutral(timestamp)
                return result("neutral", "Nenhum sinal", static_score)
            if (
                label
                and static_score >= self.thresholds["static"]
                and margin >= MIN_MARGIN
            ):
                self.static_votes.append((timestamp, label, static_score))
                while self.static_votes and timestamp - self.static_votes[0][0] > 0.8:
                    self.static_votes.popleft()
                same = [v for v in self.static_votes if v[1] == label]
                if (
                    len(same) >= 4
                    and timestamp - same[0][0] >= 0.35
                    and len(same) == len(self.static_votes)
                ):
                    candidate, score = label, float(np.mean([v[2] for v in same]))
            else:
                self.static_votes.clear()
        if candidate is None:
            return result("uncertain", score=score)
        sign = signs.get(candidate.casefold())
        if sign is None or (sign["two_hands"] and extracted["hands_detected"] != 2):
            return result("requirements", "Mostre as duas mãos", score)
        if sign["requires_body"] and not extracted["body_detected"]:
            return result("requirements", "Enquadre rosto e ombros", score)
        self.neutral_since = None
        emitted = candidate if candidate != self.emitted_label else None
        if emitted:
            self.emitted_label = candidate
            if dynamic:
                self.frames.clear()
                self.static_votes.clear()
                self.dynamic_candidate, self.dynamic_votes = None, 0
        return result("recognized", candidate, score, dynamic, emitted)
