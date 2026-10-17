"""Features v2: dedos locais, trajetória global, relação entre mãos e corpo."""

import numpy as np
from config import HAND_FEATURES_DIM, TOTAL_FEATURES_DIM, MAX_FRAME_GAP


def hand_shape(coords):
    centered = coords - coords[0]
    scale = max(float(np.linalg.norm(centered, axis=1).max()), 1e-6)
    normalized = centered / scale
    ratios = [
        np.clip(
            np.linalg.norm(normalized[t]) / (np.linalg.norm(normalized[m]) + 1e-5), 0, 3
        )
        for t, m in zip((4, 8, 12, 16, 20), (2, 5, 9, 13, 17))
    ]
    distances = [np.linalg.norm(normalized[4] - normalized[j]) for j in (8, 6, 12, 16)]
    return np.r_[1.0, normalized.ravel(), ratios, distances].astype(np.float32), scale


def assemble_features(hands, pose, aspect):
    """hands: dict Left/Right -> 21x3; pose: 5x3 (nariz, boca, ombros), ou None.

    x e z do MediaPipe usam largura como referência. y é convertido para a
    mesma unidade antes de calcular distâncias. Entrada sempre espelhada.
    """
    corrected = {
        k: np.asarray(v, np.float32) * [1.0, aspect, 1.0] for k, v in hands.items()
    }
    body = None if pose is None else np.asarray(pose, np.float32) * [1.0, aspect, 1.0]
    origin = np.array([0.5, aspect / 2, 0.0], np.float32)
    scale = 1.0
    if body is not None:
        origin = (body[3] + body[4]) / 2
        scale = max(float(np.linalg.norm((body[3] - body[4])[:2])), 0.1)
    out = np.zeros(TOTAL_FEATURES_DIM, np.float32)
    raw_motion = {}
    for index, label in enumerate(("Left", "Right")):
        if label not in corrected:
            continue
        coords = corrected[label]
        shape, hand_scale = hand_shape(coords)
        out[index * HAND_FEATURES_DIM : (index + 1) * HAND_FEATURES_DIM] = shape
        # Preservado separadamente: mover a mão agora altera as features.
        out[146 + index * 3 : 149 + index * 3] = (coords[0] - origin) / scale
        out[171 + index] = hand_scale / scale
        raw_motion[label] = coords[[0, 4, 8, 12, 16, 20], :2]
    if len(corrected) == 2:
        out[152:155] = (corrected["Right"][0] - corrected["Left"][0]) / scale
    if body is not None:
        out[155] = 1
        out[156:171] = ((body - origin) / scale).ravel()
    return out, raw_motion


class HandFeatureExtractor:
    def __init__(self):
        # Importação tardia permite diagnosticar a instalação sem derrubar o site.
        import mediapipe as mp

        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "Instale as versões do requirements.txt usando Python 3.11 ou 3.12."
            )
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2, min_detection_confidence=0.65, min_tracking_confidence=0.6
        )
        self.pose = mp.solutions.pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.previous = None
        self.last_time = None

    def reset_motion(self):
        self.previous = None
        self.last_time = None

    def reset(self):
        self.reset_motion()
        self.hands.reset()
        self.pose.reset()

    def close(self):
        self.hands.close()
        self.pose.close()

    def process_bgr_frame(self, bgr_frame, timestamp):
        import cv2

        # Convenção do modelo Hands: imagem selfie espelhada, qualquer câmera.
        rgb = cv2.cvtColor(cv2.flip(bgr_frame, 1), cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        body_result = self.pose.process(rgb)
        hands, render = {}, []
        for lms, handedness in zip(
            results.multi_hand_landmarks or [], results.multi_handedness or []
        ):
            classification = handedness.classification[0]
            label = classification.label
            if classification.score < 0.65 or label in hands:
                continue  # Não inventa lateralidade quando o rastreador está ambíguo.
            coords = np.array([[p.x, p.y, p.z] for p in lms.landmark], np.float32)
            hands[label] = coords
            # Desespelha só para desenhar sobre a imagem original recebida.
            render.append(
                {
                    "handedness": label,
                    "points": [
                        {"x": float(1 - p.x), "y": float(p.y), "z": float(p.z)}
                        for p in lms.landmark
                    ],
                }
            )
        pose = None
        if body_result.pose_landmarks:
            pts = [body_result.pose_landmarks.landmark[i] for i in (0, 9, 10, 11, 12)]
            if min(p.visibility for p in pts) >= 0.6:
                pose = np.array([[p.x, p.y, p.z] for p in pts], np.float32)
        features, motion = assemble_features(
            hands, pose, bgr_frame.shape[0] / bgr_frame.shape[1]
        )
        velocity = 0.0
        dt = timestamp - self.last_time if self.last_time is not None else None
        if self.previous is not None and dt is not None and 0 < dt <= MAX_FRAME_GAP:
            speeds = [
                np.mean(np.linalg.norm(p - self.previous[k], axis=1)) / dt
                for k, p in motion.items()
                if k in self.previous
            ]
            if speeds:
                velocity = float(np.mean(speeds))
        self.previous, self.last_time = motion, timestamp
        return {
            "features": features,
            "velocity": velocity,
            "hands_detected": len(hands),
            "body_detected": pose is not None,
            "landmarks": render,
            "tracking_signature": tuple(sorted(hands)),
        }
