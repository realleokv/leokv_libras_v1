"""Avalia o caminho completo em vídeos reais anotados, sem retreinar modelos.
Uso: py tools/evaluate_video.py manifest.json --output reports/video.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from config import DATA_DIR
from database import Database
from feature_extractor import HandFeatureExtractor
from ml_engine import LibrasMLEngine
from stabilizer import TemporalStabilizer


def score_events(expected, predictions, tolerance=0.5):
    unused = set(range(len(expected)))
    matched = []
    false = []
    for prediction in predictions:
        candidates = [
            i
            for i in unused
            if expected[i]["label"] == prediction["label"]
            and expected[i]["start"]
            <= prediction["time"]
            <= expected[i]["end"] + tolerance
        ]
        if candidates:
            index = min(
                candidates, key=lambda i: abs(expected[i]["end"] - prediction["time"])
            )
            unused.remove(index)
            matched.append(
                {
                    "label": prediction["label"],
                    "latency_from_start_s": round(
                        prediction["time"] - expected[index]["start"], 3
                    ),
                }
            )
        else:
            false.append(prediction)
    return {
        "matched": len(matched),
        "missed": len(unused),
        "false_or_repeated": len(false),
        "latencies": matched,
        "missed_events": [expected[i] for i in sorted(unused)],
        "false_events": false,
    }


def evaluate(manifest_path, data_dir=DATA_DIR):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest.get("videos"), list) or not manifest["videos"]:
        raise ValueError("Informe uma lista não vazia de vídeos.")
    db = Database(Path(data_dir) / "libras_v2.db")
    engine = LibrasMLEngine(db, Path(data_dir) / "models_v2")
    if not any(engine.available(k) for k in ("static", "dynamic")):
        raise ValueError(
            "Não há modelo v2 atualizado. Faça coleta e treinamento antes de avaliar."
        )
    signs = {s["canonical"]: s for s in db.get_signs()}
    reports = []
    for video in manifest["videos"]:
        if not isinstance(video.get("events"), list):
            raise ValueError(
                "Cada vídeo deve ter events, inclusive lista vazia para testes negativos."
            )
        for event in video["events"]:
            if (
                event["start"] < 0
                or event["end"] < event["start"]
                or event["label"].casefold() not in signs
            ):
                raise ValueError("Anotação inválida.")
        cap = cv2.VideoCapture(str(path.parent / video["file"]))
        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir {video['file']}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not 1 <= fps <= 240:
            cap.release()
            raise ValueError("FPS do vídeo inválido.")
        extractor = HandFeatureExtractor()
        stabilizer = TemporalStabilizer()
        predictions = []
        durations = []
        index = 0
        last = -100
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = index / fps
                index += 1
                if timestamp - last < 0.075:
                    continue
                last = timestamp
                start = time.perf_counter()
                extracted = extractor.process_bgr_frame(frame, timestamp)
                result = stabilizer.process_step(extracted, timestamp, engine, signs)
                durations.append(time.perf_counter() - start)
                if result["emitted"]:
                    predictions.append(
                        {"label": result["emitted"], "time": round(timestamp, 3)}
                    )
        finally:
            cap.release()
            extractor.close()
        scored = score_events(video["events"], predictions)
        scored.update(
            video=video["file"],
            duration_seconds=round(index / fps, 3),
            mean_processing_ms=round(1000 * sum(durations) / max(1, len(durations)), 2),
            predictions=predictions,
        )
        reports.append(scored)
    return {
        "scope": "Detecção, classificação, estabilização e emissão em vídeos anotados; não avalia tradução linguística.",
        "independence": "Use pessoas e sessões que não participaram do treino nem da calibração.",
        "videos": reports,
        "totals": {
            k: sum(r[k] for r in reports)
            for k in ("matched", "missed", "false_or_repeated")
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--output", default="reports/video-evaluation.json")
    args = parser.parse_args()
    try:
        report = evaluate(args.manifest, args.data_dir)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report["totals"], ensure_ascii=False))
        print(output)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Erro de avaliação: {exc}\n")
