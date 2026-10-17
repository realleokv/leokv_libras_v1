import time
from collections import deque, Counter
import numpy as np
from config import SEQ_LENGTH, DEFAULT_CONFIDENCE_THRESHOLD, MOTION_VELOCITY_THRESHOLD


class TemporalStabilizer:
    def __init__(self):
        self.static_window = deque(maxlen=8)
        self.sequence_buffer = deque(maxlen=SEQ_LENGTH)
        self.motion_history = deque(maxlen=SEQ_LENGTH)

        self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
        self.last_emitted_label = None
        self.last_emitted_time = 0.0
        self.repeat_cooldown = 2.0  # Segundos antes de aceitar a repetição do mesmo sinal

    def reset(self):
        self.static_window.clear()
        self.sequence_buffer.clear()
        self.motion_history.clear()
        self.last_emitted_label = None

    def process_step(
        self,
        features: np.ndarray,
        hands_detected: int,
        velocity: float,
        ml_engine
    ) -> dict:
        if hands_detected == 0:
            self.static_window.clear()
            self.sequence_buffer.clear()
            self.motion_history.clear()
            self.last_emitted_label = None
            return {
                "status": "no_hands",
                "display_label": "Aguardando mãos...",
                "confidence": 0.0,
                "is_dynamic": False,
                "emitted": None
            }

        has_static = ml_engine.static_model is not None
        has_dynamic = ml_engine.dynamic_model is not None

        if not has_static and not has_dynamic:
            return {
                "status": "untrained",
                "display_label": "Modelo não treinado",
                "confidence": 0.0,
                "is_dynamic": False,
                "emitted": None
            }

        self.sequence_buffer.append(features)
        self.motion_history.append(velocity)
        avg_recent_velocity = float(np.mean(list(self.motion_history)[-6:]))

        best_label = None
        best_conf = 0.0
        is_dynamic_pred = False

        # 1. Se houver movimento relevante e o buffer de sequência estiver cheio, avalia LSTM
        if has_dynamic and len(self.sequence_buffer) == SEQ_LENGTH and avg_recent_velocity >= MOTION_VELOCITY_THRESHOLD:
            seq_arr = np.array(self.sequence_buffer, dtype=np.float32)
            dyn_label, dyn_conf = ml_engine.predict_dynamic(seq_arr)
            if dyn_label and dyn_conf >= self.confidence_threshold:
                best_label = dyn_label
                best_conf = dyn_conf
                is_dynamic_pred = True

        # 2. Se a mão estiver estável (baixa velocidade) e não acionou dinâmico, avalia Random Forest Estático
        if best_label is None and has_static and avg_recent_velocity < (MOTION_VELOCITY_THRESHOLD * 1.4):
            stat_label, stat_conf = ml_engine.predict_static(features)
            if stat_label and stat_conf >= self.confidence_threshold:
                self.static_window.append((stat_label, stat_conf))
            else:
                self.static_window.append(("__LOW__", stat_conf))

            if len(self.static_window) >= 5:
                labels_in_win = [item[0] for item in self.static_window if item[0] != "__LOW__"]
                if labels_in_win:
                    most_common_lbl, count = Counter(labels_in_win).most_common(1)[0]
                    if count >= 4:
                        confs = [item[1] for item in self.static_window if item[0] == most_common_lbl]
                        best_label = most_common_lbl
                        best_conf = float(np.mean(confs))
                        is_dynamic_pred = False

        # 3. Se não atingiu confiança suficiente ou estabilidade
        if best_label is None:
            return {
                "status": "low_confidence",
                "display_label": "Sinal não reconhecido",
                "confidence": round(best_conf * 100, 1),
                "is_dynamic": False,
                "emitted": None
            }

        # 4. Controle Anti-Repetição para emissão na caixa de texto
        now = time.time()
        emitted_token = None
        if best_label != self.last_emitted_label or (now - self.last_emitted_time) > self.repeat_cooldown:
            emitted_token = best_label
            self.last_emitted_label = best_label
            self.last_emitted_time = now
            if is_dynamic_pred:
                self.sequence_buffer.clear()
                self.static_window.clear()

        return {
            "status": "recognized",
            "display_label": best_label,
            "confidence": round(best_conf * 100, 1),
            "is_dynamic": is_dynamic_pred,
            "emitted": emitted_token
        }