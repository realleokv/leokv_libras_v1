import os
import urllib.request
import threading
import cv2
import numpy as np
import mediapipe as mp
from config import HAND_FEATURES_DIM, TOTAL_FEATURES_DIM, MODELS_DIR

HAND_TASK_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
HAND_TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


class HandFeatureExtractor:
    def __init__(self):
        self.lock = threading.Lock()
        self.prev_features = None
        self.use_tasks_api = not hasattr(mp, "solutions")

        if not self.use_tasks_api:
            # Compatibilidade com versões antigas do MediaPipe (Legacy Solutions API)
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.65,
                min_tracking_confidence=0.60
            )
            print("[INFO] MediaPipe inicializado via API clássica (mp.solutions.hands).")
        else:
            # Nova MediaPipe Tasks API (Python 3.12 / 3.13 / MediaPipe 0.10.22+)
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            self._ensure_task_model_downloaded()

            # Lê em bytes (model_asset_buffer) para evitar erro de acentuação no caminho do Windows ("Área de Trabalho")
            with open(HAND_TASK_MODEL_PATH, "rb") as f:
                model_bytes = f.read()

            base_options = mp_python.BaseOptions(model_asset_buffer=model_bytes)
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.60,
                min_hand_presence_confidence=0.60
            )
            self.detector = mp_vision.HandLandmarker.create_from_options(options)
            print("[INFO] MediaPipe inicializado via nova Tasks API (HandLandmarker).")

    @staticmethod
    def _ensure_task_model_downloaded():
        os.makedirs(MODELS_DIR, exist_ok=True)
        if not os.path.exists(HAND_TASK_MODEL_PATH):
            print("[INFO] Baixando modelo oficial 'hand_landmarker.task' (~12 MB) do Google...")
            urllib.request.urlretrieve(HAND_TASK_MODEL_URL, HAND_TASK_MODEL_PATH)
            print("[INFO] Download do 'hand_landmarker.task' concluído com sucesso!")

    @staticmethod
    def _extract_single_hand_features(landmarks) -> np.ndarray:
        """
        Extrai 73 features invariantes à translação e escala para uma única mão:
        - 1 flag de presença (1.0)
        - 63 coordenadas 3D normalizadas em relação ao pulso (landmark 0)
        - 5 razões de extensão dos dedos (ponta vs MCP em relação ao pulso)
        - 4 distâncias críticas do polegar para desambiguação (A/S/E/M/N/T/F)
        """
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

        # Centraliza no pulso (landmark 0)
        wrist = coords[0].copy()
        centered = coords - wrist

        # Escala pela distância máxima ao pulso (invariância à distância da câmera)
        distances = np.linalg.norm(centered, axis=1)
        max_dist = np.max(distances)
        if max_dist < 1e-6:
            max_dist = 1.0
        normalized = centered / max_dist
        flat_coords = normalized.flatten()  # 63 valores

        # 5 razões de extensão dos dedos: (dist ponta ao pulso) / (dist MCP ao pulso)
        tips = [4, 8, 12, 16, 20]
        mcps = [2, 5, 9, 13, 17]
        extension_ratios = []
        for tip_idx, mcp_idx in zip(tips, mcps):
            d_tip = np.linalg.norm(normalized[tip_idx])
            d_mcp = np.linalg.norm(normalized[mcp_idx]) + 1e-5
            extension_ratios.append(np.clip(d_tip / d_mcp, 0.0, 3.0))

        # 4 distâncias entre a ponta do polegar (4) e pontos-chave (8, 6, 12, 16)
        thumb_tip = normalized[4]
        key_targets = [8, 6, 12, 16]
        thumb_distances = [
            float(np.linalg.norm(thumb_tip - normalized[idx]))
            for idx in key_targets
        ]

        features = np.concatenate([
            np.array([1.0], dtype=np.float32),
            flat_coords,
            np.array(extension_ratios, dtype=np.float32),
            np.array(thumb_distances, dtype=np.float32)
        ])
        return features  # Dimensão: 73

    def process_bgr_frame(self, bgr_frame: np.ndarray) -> dict:
        """
        Processa um frame BGR do OpenCV e retorna:
        - features: np.ndarray de 146 posições (Left 73 + Right 73)
        - hands_detected: int
        - velocity: float (velocidade média dos pontos entre frames)
        - landmarks: lista serializável para desenhar no canvas do frontend
        """
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame)

        left_slot = np.zeros(HAND_FEATURES_DIM, dtype=np.float32)
        right_slot = np.zeros(HAND_FEATURES_DIM, dtype=np.float32)
        landmarks_render = []
        hands_detected = 0

        with self.lock:
            if not self.use_tasks_api:
                rgb_frame.flags.writeable = False
                results = self.hands.process(rgb_frame)
                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        hands_detected += 1
                        label = handedness.classification[0].label
                        score = float(handedness.classification[0].score)
                        lm_list = hand_lms.landmark

                        single_feat = self._extract_single_hand_features(lm_list)
                        if label == "Left" and left_slot[0] == 0.0:
                            left_slot = single_feat
                        elif label == "Right" and right_slot[0] == 0.0:
                            right_slot = single_feat
                        elif left_slot[0] == 0.0:
                            left_slot = single_feat
                        else:
                            right_slot = single_feat

                        landmarks_render.append({
                            "handedness": label,
                            "score": round(score, 3),
                            "points": [
                                {"x": round(lm.x, 4), "y": round(lm.y, 4), "z": round(lm.z, 4)}
                                for lm in lm_list
                            ]
                        })
            else:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = self.detector.detect(mp_image)

                if detection_result.hand_landmarks and detection_result.handedness:
                    for lm_list, handedness_list in zip(detection_result.hand_landmarks, detection_result.handedness):
                        hands_detected += 1
                        top_hand = handedness_list[0]
                        label = getattr(top_hand, "category_name", None) or getattr(top_hand, "display_name", "Right")
                        score = float(getattr(top_hand, "score", 1.0))

                        single_feat = self._extract_single_hand_features(lm_list)
                        if label == "Left" and left_slot[0] == 0.0:
                            left_slot = single_feat
                        elif label == "Right" and right_slot[0] == 0.0:
                            right_slot = single_feat
                        elif left_slot[0] == 0.0:
                            left_slot = single_feat
                        else:
                            right_slot = single_feat

                        landmarks_render.append({
                            "handedness": label,
                            "score": round(score, 3),
                            "points": [
                                {"x": round(lm.x, 4), "y": round(lm.y, 4), "z": round(lm.z, 4)}
                                for lm in lm_list
                            ]
                        })

        full_features = np.concatenate([left_slot, right_slot])
        assert full_features.shape[0] == TOTAL_FEATURES_DIM

        # Cálculo de velocidade de movimento entre frames consecutivos
        velocity = 0.0
        if hands_detected > 0 and self.prev_features is not None:
            active_mask = (full_features[0] > 0 and self.prev_features[0] > 0) or (
                full_features[HAND_FEATURES_DIM] > 0 and self.prev_features[HAND_FEATURES_DIM] > 0
            )
            if active_mask:
                diff = np.abs(full_features - self.prev_features)
                velocity = float(np.mean(diff))

        self.prev_features = full_features.copy() if hands_detected > 0 else None

        return {
            "features": full_features,
            "hands_detected": hands_detected,
            "velocity": velocity,
            "landmarks": landmarks_render
        }