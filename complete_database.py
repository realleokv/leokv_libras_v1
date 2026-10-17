import os
import math
import numpy as np
from types import SimpleNamespace

from config import (
    STATIC_DATA_PATH,
    DYNAMIC_DATA_PATH,
    HAND_FEATURES_DIM,
    TOTAL_FEATURES_DIM,
    SEQ_LENGTH,
    DEFAULT_SIGNS_CATALOG
)
from database import init_db
from feature_extractor import HandFeatureExtractor
from ml_engine import LibrasMLEngine


def create_lm(x: float, y: float, z: float):
    return SimpleNamespace(x=float(x), y=float(y), z=float(z))


def rotate_points_3d(pts: np.ndarray, rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Aplica matriz de rotação 3D Euler (Rx, Ry, Rz) nos 21 pontos da mão."""
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)

    R = Rz @ Ry @ Rx
    return pts @ R.T


def build_anatomical_hand(
    curl_factors: list,
    spread_angles: list,
    thumb_cfg: tuple,
    hand_rot: tuple = (0.0, 0.0, 0.0),
    cross_index_middle: bool = False
) -> np.ndarray:
    """
    Constrói os 21 landmarks 3D anatômicos no referencial do MediaPipe:
    - Pulso (0) em (0, 0, 0)
    - Eixo Y cresce para baixo na tela (dedos apontando para cima têm Y negativo)
    - curl_factors: [indicador, médio, anelar, mínimo] de 0.0 (estendido) a 1.0 (fechado em punho)
    - spread_angles: abertura lateral em graus [indicador, médio, anelar, mínimo]
    - thumb_cfg: (x_tip, y_tip, z_tip) posição alvo da ponta do polegar relativa à palma
    """
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.0, 0.0, 0.0]  # Pulso

    # Bases metacarpais (MCPs: 5=Indicador, 9=Médio, 13=Anelar, 17=Mínimo)
    mcp_bases = [
        np.array([-0.22, -0.42, 0.0], dtype=np.float32),
        np.array([-0.07, -0.45, 0.0], dtype=np.float32),
        np.array([0.08, -0.43, 0.0], dtype=np.float32),
        np.array([0.22, -0.38, 0.0], dtype=np.float32),
    ]
    finger_lengths = [
        [0.20, 0.14, 0.10],  # Indicador (6, 7, 8)
        [0.22, 0.16, 0.11],  # Médio (10, 11, 12)
        [0.20, 0.15, 0.10],  # Anelar (14, 15, 16)
        [0.16, 0.11, 0.09],  # Mínimo (18, 19, 20)
    ]
    mcp_indices = [5, 9, 13, 17]

    # Construção cinemática do Polegar (1=CMC, 2=MCP, 3=IP, 4=TIP)
    tx, ty, tz = thumb_cfg
    pts[1] = [-0.18, -0.14, -0.03]
    pts[2] = [-0.28, -0.26, -0.05]
    pts[3] = pts[2] + 0.55 * (np.array([tx, ty, tz], dtype=np.float32) - pts[2])
    pts[4] = [tx, ty, tz]

    # Construção cinemática dos 4 dedos (Indicador ao Mínimo)
    for f_idx in range(4):
        mcp_idx = mcp_indices[f_idx]
        base = mcp_bases[f_idx]
        pts[mcp_idx] = base

        curl = float(curl_factors[f_idx])
        spread_rad = math.radians(float(spread_angles[f_idx]))
        lengths = finger_lengths[f_idx]

        # Direção inicial apontando para cima (Y negativo) com abertura lateral
        dir_x = math.sin(spread_rad)
        dir_y = -math.cos(spread_rad)

        # Se curl <= 0.5 (garra/curva C/E) vs curl > 0.5 (dobrado sobre a palma)
        curr = base.copy()
        for j_step, seg_len in enumerate(lengths):
            joint_idx = mcp_idx + 1 + j_step
            # Ângulo de flexão progressivo por falange
            flex_angle = curl * math.pi * (0.35 + 0.32 * j_step)
            dy = dir_y * math.cos(flex_angle)
            dz = -math.sin(flex_angle) * 0.65
            dx = dir_x * math.cos(flex_angle * 0.5)

            step_vec = np.array([dx, dy, dz], dtype=np.float32) * seg_len
            curr = curr + step_vec
            pts[joint_idx] = curr

    # Caso especial Letra R (cruzar indicador e médio)
    if cross_index_middle:
        pts[7][0] += 0.09
        pts[8][0] += 0.14
        pts[11][0] -= 0.05
        pts[12][0] -= 0.08
        pts[8][2] -= 0.04

    # Aplica orientação global da mão (ex.: M, N, P, Q apontam para baixo/frente)
    pts = rotate_points_3d(pts, hand_rot[0], hand_rot[1], hand_rot[2])
    return pts


# Presets biomecânicos das configurações de mão da Libras
HAND_CONFIGS = {
    "A": dict(curl=[0.95, 0.95, 0.95, 0.95], spread=[0, 0, 0, 0], thumb=(-0.27, -0.48, -0.04), rot=(0, 0, 0)),
    "B": dict(curl=[0.0, 0.0, 0.0, 0.0], spread=[-2, 0, 2, 4], thumb=(0.02, -0.32, -0.08), rot=(0, 0, 0)),
    "C": dict(curl=[0.42, 0.42, 0.42, 0.42], spread=[0, 2, 4, 6], thumb=(-0.22, -0.52, -0.18), rot=(0, 25, 0)),
    "D": dict(curl=[0.0, 0.72, 0.75, 0.78], spread=[0, 0, 0, 0], thumb=(-0.05, -0.52, -0.14), rot=(0, 0, 0)),
    "E": dict(curl=[0.62, 0.62, 0.62, 0.62], spread=[0, 0, 0, 0], thumb=(-0.06, -0.34, -0.10), rot=(0, 0, 0)),
    "F": dict(curl=[0.48, 0.0, 0.0, 0.0], spread=[-4, 0, 3, 6], thumb=(-0.26, -0.58, -0.14), rot=(0, 0, 0)),
    "G": dict(curl=[0.0, 0.92, 0.92, 0.92], spread=[0, 0, 0, 0], thumb=(-0.28, -0.68, -0.04), rot=(0, 0, -15)),
    "H": dict(curl=[0.0, 0.0, 0.92, 0.92], spread=[-8, 8, 0, 0], thumb=(-0.14, -0.58, -0.08), rot=(0, 0, 0)),
    "I": dict(curl=[0.92, 0.92, 0.92, 0.0], spread=[0, 0, 0, 6], thumb=(-0.10, -0.42, -0.09), rot=(0, 0, 0)),
    "J": dict(curl=[0.92, 0.92, 0.92, 0.0], spread=[0, 0, 0, 6], thumb=(-0.10, -0.42, -0.09), rot=(0, 0, 0)),
    "K": dict(curl=[0.0, 0.22, 0.92, 0.92], spread=[-6, 8, 0, 0], thumb=(-0.14, -0.56, -0.10), rot=(0, 0, 0)),
    "L": dict(curl=[0.0, 0.92, 0.92, 0.92], spread=[0, 0, 0, 0], thumb=(-0.48, -0.34, -0.02), rot=(0, 0, 0)),
    "M": dict(curl=[0.05, 0.05, 0.05, 0.88], spread=[-4, 0, 4, 0], thumb=(0.12, -0.38, -0.08), rot=(165, 0, 0)),
    "N": dict(curl=[0.05, 0.05, 0.90, 0.90], spread=[-4, 4, 0, 0], thumb=(0.06, -0.38, -0.08), rot=(165, 0, 0)),
    "O": dict(curl=[0.58, 0.58, 0.58, 0.58], spread=[-4, -1, 2, 5], thumb=(-0.12, -0.55, -0.16), rot=(0, 20, 0)),
    "P": dict(curl=[0.0, 0.25, 0.92, 0.92], spread=[-5, 6, 0, 0], thumb=(-0.14, -0.55, -0.10), rot=(85, 20, -65)),
    "Q": dict(curl=[0.0, 0.92, 0.92, 0.92], spread=[0, 0, 0, 0], thumb=(-0.25, -0.65, -0.05), rot=(170, 0, 0)),
    "R": dict(curl=[0.0, 0.0, 0.92, 0.92], spread=[0, 0, 0, 0], thumb=(0.04, -0.40, -0.09), rot=(0, 0, 0), cross=True),
    "S": dict(curl=[0.95, 0.95, 0.95, 0.95], spread=[0, 0, 0, 0], thumb=(-0.05, -0.44, -0.14), rot=(0, 0, 0)),
    "T": dict(curl=[0.48, 0.0, 0.0, 0.0], spread=[-4, 4, 6, 8], thumb=(-0.12, -0.56, -0.06), rot=(0, 0, 0)),
    "U": dict(curl=[0.0, 0.0, 0.92, 0.92], spread=[-1, 1, 0, 0], thumb=(0.05, -0.40, -0.09), rot=(0, 0, 0)),
    "V": dict(curl=[0.0, 0.0, 0.92, 0.92], spread=[-16, 16, 0, 0], thumb=(0.05, -0.40, -0.09), rot=(0, 0, 0)),
    "W": dict(curl=[0.0, 0.0, 0.0, 0.92], spread=[-18, 0, 18, 0], thumb=(0.14, -0.38, -0.09), rot=(0, 0, 0)),
    "X": dict(curl=[0.52, 0.92, 0.92, 0.92], spread=[0, 0, 0, 0], thumb=(-0.12, -0.42, -0.09), rot=(0, 15, 0)),
    "Y": dict(curl=[0.92, 0.92, 0.92, 0.0], spread=[0, 0, 0, 18], thumb=(-0.46, -0.35, -0.02), rot=(0, 0, 0)),
    "Z": dict(curl=[0.0, 0.92, 0.92, 0.92], spread=[0, 0, 0, 0], thumb=(-0.12, -0.42, -0.09), rot=(0, 0, -8)),
    # Palavras Estáticas
    "Casa": dict(curl=[0.0, 0.0, 0.0, 0.0], spread=[-1, 0, 1, 2], thumb=(-0.18, -0.42, -0.04), rot=(0, 15, 32), two_hands=True),
    "Eu": dict(curl=[0.0, 0.92, 0.92, 0.92], spread=[0, 0, 0, 0], thumb=(-0.12, -0.42, -0.09), rot=(25, -45, 75)),
    "Você": dict(curl=[0.0, 0.92, 0.92, 0.92], spread=[0, 0, 0, 0], thumb=(-0.12, -0.42, -0.09), rot=(70, 0, 0)),
}


def extract_73_from_pts(pts: np.ndarray, mirror_x: bool = False) -> np.ndarray:
    pts_mod = pts.copy()
    if mirror_x:
        pts_mod[:, 0] *= -1.0
    lm_objs = [create_lm(p[0], p[1], p[2]) for p in pts_mod]
    return HandFeatureExtractor._extract_single_hand_features(lm_objs)


def generate_static_feature_vector(label: str, rng: np.random.Generator, slot_mode: str = "right") -> np.ndarray:
    cfg = HAND_CONFIGS.get(label, HAND_CONFIGS["B"])

    # Microvariações angulares e anatômicas realistas
    curl_jitter = np.clip(np.array(cfg["curl"]) + rng.normal(0, 0.025, 4), 0.0, 1.0)
    spread_jitter = np.array(cfg["spread"]) + rng.normal(0, 1.5, 4)
    thumb_jitter = tuple(np.array(cfg["thumb"]) + rng.normal(0, 0.012, 3))
    rot_jitter = (
        cfg["rot"][0] + rng.uniform(-9, 9),
        cfg["rot"][1] + rng.uniform(-9, 9),
        cfg["rot"][2] + rng.uniform(-9, 9),
    )

    pts = build_anatomical_hand(
        curl_factors=curl_jitter,
        spread_angles=spread_jitter,
        thumb_cfg=thumb_jitter,
        hand_rot=rot_jitter,
        cross_index_middle=cfg.get("cross", False)
    )
    pts += rng.normal(0, 0.006, pts.shape).astype(np.float32)

    if cfg.get("two_hands", False) or slot_mode == "both":
        left_f = extract_73_from_pts(pts, mirror_x=True)
        right_f = extract_73_from_pts(pts, mirror_x=False)
        return np.concatenate([left_f, right_f])
    elif slot_mode == "left":
        left_f = extract_73_from_pts(pts, mirror_x=True)
        return np.concatenate([left_f, np.zeros(HAND_FEATURES_DIM, dtype=np.float32)])
    else:
        right_f = extract_73_from_pts(pts, mirror_x=False)
        return np.concatenate([np.zeros(HAND_FEATURES_DIM, dtype=np.float32), right_f])


# Trajetórias Temporais de 20 frames para Sinais Dinâmicos (Início -> Fim + Rotação/Translação)
DYNAMIC_SPECS = {
    "H": ("H", "H", (0, 0, -35), (0, 55, 35), False),
    "J": ("I", "I", (0, 0, -10), (35, 30, 45), False),
    "K": ("K", "K", (15, 0, 0), (-25, 0, 0), False),
    "X": ("X", "X", (-15, 10, 0), (25, 35, -15), False),
    "Z": ("Z", "Z", (-15, -20, -25), (20, 25, 30), False),
    "Oi": ("O", "I", (0, 15, -10), (0, -15, 20), False),
    "Olá": ("B", "B", (0, 0, -28), (0, 0, 28), False),
    "Bom dia": ("O", "D", (10, 0, 0), (0, 15, -15), False),
    "Boa tarde": ("B", "B", (-10, 0, 15), (35, 15, -20), True),
    "Boa noite": ("B", "C", (-15, 10, 0), (30, -10, 15), True),
    "Tudo bem?": ("O", "A", (5, 0, 0), (-15, 20, -25), False),
    "Meu nome": ("B", "U", (20, -25, 60), (0, 20, -70), False),
    "Obrigado": ("B", "B", (-20, 0, 0), (45, 10, -10), False),
    "Por favor": ("B", "B", (0, 25, 30), (15, -15, -10), True),
    "Desculpa": ("Y", "Y", (-15, 10, 0), (18, -10, 12), False),
    "Sim": ("S", "S", (-25, 0, 0), (45, 0, 0), False),
    "Não": ("D", "D", (0, -25, -30), (0, 25, 30), False),
    "Ajuda": ("A", "B", (15, 0, 0), (-25, 20, 10), True),
    "Água": ("L", "L", (0, 0, -8), (22, 0, 25), False),
    "Comida": ("O", "E", (25, 10, 0), (-20, -10, 15), False),
    "Escola": ("Casa", "B", (0, 15, 25), (15, -15, -20), True),
}


def generate_dynamic_sequence(label: str, rng: np.random.Generator, slot_mode: str = "right") -> np.ndarray:
    start_key, end_key, rot_start, rot_end, two_hands = DYNAMIC_SPECS.get(
        label, ("B", "B", (0, 0, -20), (0, 0, 20), False)
    )
    cfg_s = HAND_CONFIGS.get(start_key, HAND_CONFIGS["B"])
    cfg_e = HAND_CONFIGS.get(end_key, HAND_CONFIGS["B"])

    seq_frames = []
    for t_idx in range(SEQ_LENGTH):
        alpha = t_idx / float(SEQ_LENGTH - 1)
        # Curva harmônica para diferenciar movimentos cíclicos e traçados (J, Z, Olá, Não)
        wave = math.sin(alpha * math.pi * 2.0) * 12.0 if label in ("Z", "Olá", "Não", "Água", "Comida") else 0.0

        curl_t = (1.0 - alpha) * np.array(cfg_s["curl"]) + alpha * np.array(cfg_e["curl"])
        curl_t = np.clip(curl_t + rng.normal(0, 0.015, 4), 0.0, 1.0)

        spread_t = (1.0 - alpha) * np.array(cfg_s["spread"]) + alpha * np.array(cfg_e["spread"])
        thumb_t = tuple((1.0 - alpha) * np.array(cfg_s["thumb"]) + alpha * np.array(cfg_e["thumb"]))

        rx = (1.0 - alpha) * rot_start[0] + alpha * rot_end[0] + rng.normal(0, 2.0)
        ry = (1.0 - alpha) * rot_start[1] + alpha * rot_end[1] + rng.normal(0, 2.0)
        rz = (1.0 - alpha) * rot_start[2] + alpha * rot_end[2] + wave + rng.normal(0, 2.0)

        pts = build_anatomical_hand(curl_t, spread_t, thumb_t, (rx, ry, rz))
        pts += rng.normal(0, 0.004, pts.shape).astype(np.float32)

        if two_hands:
            lf = extract_73_from_pts(pts, mirror_x=True)
            rf = extract_73_from_pts(pts, mirror_x=False)
            frame_vec = np.concatenate([lf, rf])
        elif slot_mode == "left":
            lf = extract_73_from_pts(pts, mirror_x=True)
            frame_vec = np.concatenate([lf, np.zeros(HAND_FEATURES_DIM, dtype=np.float32)])
        else:
            rf = extract_73_from_pts(pts, mirror_x=False)
            frame_vec = np.concatenate([np.zeros(HAND_FEATURES_DIM, dtype=np.float32), rf])

        seq_frames.append(frame_vec)

    return np.array(seq_frames, dtype=np.float32)


def augment_existing_static_vector(vec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Aplica perturbação proporcional apenas nas mãos ativas de uma amostra real já gravada."""
    out = vec.copy()
    for offset in (0, HAND_FEATURES_DIM):
        if out[offset] > 0.5:
            noise = rng.normal(0, 0.015, HAND_FEATURES_DIM - 1).astype(np.float32)
            out[offset + 1: offset + HAND_FEATURES_DIM] += noise
    return out


def main():
    print("=" * 68)
    print(" GERADOR MATEMÁTICO & AUGMENTATION 3D DE COORDENADAS LIBRAS ")
    print("=" * 68)

    init_db()
    rng = np.random.default_rng(42)

    # 1. Carrega amostras estáticas existentes
    if os.path.exists(STATIC_DATA_PATH):
        data_s = np.load(STATIC_DATA_PATH, allow_pickle=True)
        X_s, y_s = list(data_s["X"]), list(data_s["y"])
        print(f"[INFO] Amostras estáticas reais encontradas em '{STATIC_DATA_PATH}': {len(y_s)}")
    else:
        X_s, y_s = [], []

    # 2. Completa sinais estáticos até TARGET_STATIC amostras por classe (mão direita + esquerda)
    TARGET_STATIC = 60
    static_catalog = [s for s in DEFAULT_SIGNS_CATALOG if s["is_dynamic"] == 0]

    for item in static_catalog:
        lbl = item["label"]
        existing_indices = [i for i, y_val in enumerate(y_s) if str(y_val) == lbl]
        current_count = len(existing_indices)

        if current_count >= TARGET_STATIC:
            continue

        needed = TARGET_STATIC - current_count
        for k in range(needed):
            # Se o usuário já gravou amostras reais desse sinal, multiplica 50% a partir das gravações reais
            if current_count > 0 and k % 2 == 0:
                base_vec = X_s[rng.choice(existing_indices)]
                X_s.append(augment_existing_static_vector(base_vec, rng))
            else:
                slot = "both" if item["two_hands"] else ("right" if k % 2 == 0 else "left")
                X_s.append(generate_static_feature_vector(lbl, rng, slot_mode=slot))
            y_s.append(lbl)

    X_s_arr = np.array(X_s, dtype=np.float32)
    y_s_arr = np.array(y_s, dtype=str)
    np.savez_compressed(STATIC_DATA_PATH, X=X_s_arr, y=y_s_arr)
    print(f"[OK] Dataset estático salvo em '{STATIC_DATA_PATH}' com {len(y_s_arr)} amostras ({len(static_catalog)} classes).")

    # 3. Carrega sequências dinâmicas existentes
    if os.path.exists(DYNAMIC_DATA_PATH):
        data_d = np.load(DYNAMIC_DATA_PATH, allow_pickle=True)
        X_d, y_d = list(data_d["X"]), list(data_d["y"])
        print(f"[INFO] Sequências dinâmicas reais encontradas em '{DYNAMIC_DATA_PATH}': {len(y_d)}")
    else:
        X_d, y_d = [], []

    # 4. Completa sinais dinâmicos até TARGET_DYNAMIC sequências de 20 frames por classe
    TARGET_DYNAMIC = 24
    dynamic_catalog = [s for s in DEFAULT_SIGNS_CATALOG if s["is_dynamic"] == 1]

    for item in dynamic_catalog:
        lbl = item["label"]
        existing_indices = [i for i, y_val in enumerate(y_d) if str(y_val) == lbl]
        current_count = len(existing_indices)

        if current_count >= TARGET_DYNAMIC:
            continue

        needed = TARGET_DYNAMIC - current_count
        for k in range(needed):
            slot = "both" if item["two_hands"] else ("right" if k % 2 == 0 else "left")
            seq = generate_dynamic_sequence(lbl, rng, slot_mode=slot)
            X_d.append(seq)
            y_d.append(lbl)

    X_d_arr = np.array(X_d, dtype=np.float32)
    y_d_arr = np.array(y_d, dtype=str)
    np.savez_compressed(DYNAMIC_DATA_PATH, X=X_d_arr, y=y_d_arr)
    print(f"[OK] Dataset dinâmico salvo em '{DYNAMIC_DATA_PATH}' com {len(y_d_arr)} sequências ({len(dynamic_catalog)} classes).")

    # 5. Sincroniza banco SQLite e treina automaticamente os dois modelos
    print("\n[INFO] Treinando modelos Scikit-learn e Temporal com o dataset completo...")
    engine = LibrasMLEngine()
    res_static = engine.train_static_model()
    res_dynamic = engine.train_dynamic_model()

    print(f" -> {res_static['message']}")
    print(f" -> {res_dynamic['message']}")
    print("=" * 68)
    print(" CONCLUÍDO! Agora basta rodar: py app.py --https ")
    print("=" * 68)


if __name__ == "__main__":
    main()