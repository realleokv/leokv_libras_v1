"""Treino agrupado por pessoa, calibração separada e publicação atômica."""

import hashlib
import json
import os
import threading
import uuid
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import FEATURE_VERSION, TOTAL_FEATURES_DIM, BACKGROUND_LABEL, MIN_MARGIN
from database import utc_now
from temporal import resample_sequence


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def grouped_partitions(records):
    """Nenhuma pessoa, sessão ou gravação atravessa treino, calibração e teste."""
    labels = np.array([r["display_label"] for r in records])
    people = np.array([r["signer"] for r in records])
    classes = set(labels)
    if len(classes) < 3 or BACKGROUND_LABEL not in classes:
        raise ValueError(
            "Colete pelo menos dois sinais e a classe Nenhum sinal neste modo."
        )
    unique_people = np.unique(people)
    for label in sorted(classes):
        mask = labels == label
        if mask.sum() < 6 or len(set(people[mask])) < 3:
            raise ValueError(
                f"{label}: são necessárias 6 gravações reais de pelo menos 3 pessoas."
            )
    rng = np.random.default_rng(42)
    size = max(1, round(len(unique_people) * 0.2))
    for _ in range(512):
        perm = rng.permutation(unique_people)
        groups = (perm[2 * size :], perm[:size], perm[size : 2 * size])
        parts = [np.flatnonzero(np.isin(people, g)) for g in groups]
        if all(set(labels[p]) == classes for p in parts):
            return parts
    raise ValueError(
        "Não há cobertura de todas as classes nos três grupos. Grave todos os sinais com as mesmas 3 ou mais pessoas."
    )


def fit_temperature(logits, targets):
    def loss(temp):
        p = softmax(logits / temp, axis=1)
        return float(
            -np.log(np.clip(p[np.arange(len(targets)), targets], 1e-9, 1)).mean()
        )

    return float(minimize_scalar(loss, bounds=(0.5, 5.0), method="bounded").x)


def make_lstm(classes):
    import torch.nn as nn

    class LibrasLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(TOTAL_FEATURES_DIM, 64, batch_first=True)
            self.fc = nn.Linear(64, classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1])

    return LibrasLSTM()


class LibrasMLEngine:
    def __init__(self, database, model_dir):
        self.db, self.directory = database, Path(model_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.models = {}
        self.errors = {}
        self.load_models()

    def load_models(self):
        registry = self.directory / "registry.json"
        if not registry.exists():
            return
        try:
            entries = json.loads(registry.read_text(encoding="utf-8"))
            for kind, folder in entries.items():
                if (
                    kind not in ("static", "dynamic")
                    or len(folder) != 32
                    or any(c not in "0123456789abcdef" for c in folder)
                ):
                    raise ValueError("Registro de modelos inválido.")
                try:
                    root = self.directory / folder
                    meta = json.loads(
                        (root / "metadata.json").read_text(encoding="utf-8")
                    )
                    if (
                        meta["feature_version"] != FEATURE_VERSION
                        or meta["features_dim"] != TOTAL_FEATURES_DIM
                    ):
                        raise ValueError(
                            "Modelo de outra versão de features. Faça nova coleta e treinamento."
                        )
                    if meta["revision"] != self.db.revision(kind):
                        raise ValueError(
                            "Amostras alteradas desde o treinamento. Treine novamente."
                        )
                    model_path = root / (
                        "model.skops" if kind == "static" else "model.pt"
                    )
                    if (
                        hashlib.sha256(model_path.read_bytes()).hexdigest()
                        != meta["sha256"]
                    ):
                        raise ValueError("Arquivo de modelo danificado.")
                    if kind == "static":
                        import skops.io as sio

                        # Não aceita tipos arbitrários trazidos por arquivos externos.
                        model = sio.load(model_path, trusted=[])
                    else:
                        import torch

                        model = make_lstm(len(meta["classes"]))
                        model.load_state_dict(
                            torch.load(
                                model_path, map_location="cpu", weights_only=True
                            )
                        )
                        model.eval()
                    self.models[kind] = (model, meta)
                except Exception as e:
                    self.errors[kind] = str(e)
        except Exception as e:
            self.errors["registry"] = str(e)

    def invalidate(self, kind):
        with self.lock:
            self.models.pop(kind, None)
            self.errors[kind] = (
                "Dados alterados. Modelo desativado até novo treinamento."
            )

    def status(self):
        summaries = self.db.summaries()
        result = {}
        with self.lock:
            for kind in ("static", "dynamic"):
                loaded = self.models.get(kind)
                metadata = loaded[1] if loaded else summaries.get(kind, {})
                ready = loaded is not None and metadata.get(
                    "revision"
                ) == self.db.revision(kind)
                result[kind] = {
                    "ready": ready,
                    "state": "pronto" if ready else "não disponível",
                    "error": self.errors.get(kind),
                    "classes": metadata.get("classes", []) if ready else [],
                    "metrics": metadata if ready else None,
                }
            result["errors"] = self.errors.copy()
        return result

    def train(self, kind, progress, epochs=35):
        if kind not in ("static", "dynamic"):
            raise ValueError("Tipo de modelo inválido.")
        records = self.db.recordings(kind)
        train, calibration, test = grouped_partitions(records)
        classes = sorted({r["display_label"] for r in records})
        label_idx = {label: i for i, label in enumerate(classes)}
        y = np.array([label_idx[r["display_label"]] for r in records])
        revision = self.db.revision(kind)
        progress(f"{kind}: treinando com pessoas separadas")
        extras = {}
        if kind == "static":
            # Limita o peso de cada gravação; todas as partições são definidas antes dos frames.
            frames = [
                r["features"][
                    np.linspace(
                        0, len(r["features"]) - 1, min(30, len(r["features"]))
                    ).astype(int)
                ]
                for r in records
            ]
            model = RandomForestClassifier(
                n_estimators=160,
                max_depth=18,
                class_weight="balanced",
                random_state=42,
                n_jobs=2,
            )
            model.fit(
                np.concatenate([frames[i] for i in train]),
                np.concatenate([np.repeat(y[i], len(frames[i])) for i in train]),
            )

            def logits_for(indices):
                probabilities = np.stack(
                    [model.predict_proba(frames[i]).mean(axis=0) for i in indices]
                )
                return np.log(np.clip(probabilities, 1e-7, 1))

        else:
            import torch

            torch.manual_seed(42)
            torch.set_num_threads(2)
            sequences = np.stack(
                [resample_sequence(r["features"], r["timestamps"]) for r in records]
            )
            mean = sequences[train].mean(axis=(0, 1))
            scale = np.maximum(sequences[train].std(axis=(0, 1)), 0.05)
            normalized = (sequences - mean) / scale
            extras = {"mean": mean.tolist(), "scale": scale.tolist()}
            model = make_lstm(len(classes))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            weights = len(train) / (
                len(classes) * np.bincount(y[train], minlength=len(classes))
            )
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(weights, dtype=torch.float32)
            )
            gen = np.random.default_rng(42)
            model.train()
            for epoch in range(epochs):
                order = gen.permutation(train)
                for start in range(0, len(order), 16):
                    idx = order[start : start + 16]
                    optimizer.zero_grad()
                    loss = loss_fn(
                        model(torch.tensor(normalized[idx])),
                        torch.tensor(y[idx], dtype=torch.long),
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                progress(f"Dinâmico: época {epoch+1}/{epochs}")
            model.eval()

            def logits_for(indices):
                with torch.no_grad():
                    return model(
                        torch.tensor(normalized[indices], dtype=torch.float32)
                    ).numpy()

        temperature = fit_temperature(logits_for(calibration), y[calibration])
        probs = softmax(logits_for(test) / temperature, axis=1)
        preds = probs.argmax(axis=1)
        report = classification_report(
            y[test],
            preds,
            labels=list(range(len(classes))),
            target_names=classes,
            output_dict=True,
            zero_division=0,
        )
        meta = {
            "kind": kind,
            "feature_version": FEATURE_VERSION,
            "features_dim": TOTAL_FEATURES_DIM,
            "revision": revision,
            "classes": classes,
            "temperature": temperature,
            "accuracy": float(accuracy_score(y[test], preds)),
            "report": report,
            "confusion_matrix": confusion_matrix(
                y[test], preds, labels=list(range(len(classes)))
            ).tolist(),
            "recordings": len(records),
            "test_recordings": len(test),
            "people": len({r["signer"] for r in records}),
            "split_people_counts": [
                len({records[i]["signer"] for i in p})
                for p in (train, calibration, test)
            ],
            "evaluation": "Gravações reais separadas por pessoa; não é avaliação de tradução contínua.",
            "updated_at": utc_now(),
            **extras,
        }
        # Não retreina no conjunto de teste depois de medir o desempenho.
        folder = uuid.uuid4().hex
        root = self.directory / folder
        root.mkdir()
        path = root / ("model.skops" if kind == "static" else "model.pt")
        if kind == "static":
            import skops.io as sio

            sio.dump(model, path)
        else:
            import torch

            torch.save(model.state_dict(), path)
        meta["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        atomic_json(root / "metadata.json", meta)
        if revision != self.db.revision(kind):
            raise ValueError("Dados mudaram durante o treino. Modelo não publicado.")
        with self.lock:
            registry = self.directory / "registry.json"
            entries = (
                json.loads(registry.read_text(encoding="utf-8"))
                if registry.exists()
                else {}
            )
            entries[kind] = folder
            atomic_json(registry, entries)
            self.db.set_summary(kind, meta)
            self.models[kind] = (model, meta)
            self.errors.pop(kind, None)
        progress(f"{kind}: concluído")
        return meta

    def available(self, kind):
        with self.lock:
            return kind in self.models

    def predict(self, kind, features):
        with self.lock:
            bundle = self.models.get(kind)
            if bundle is None:
                return None, 0.0, 0.0
            model, meta = bundle
            if kind == "static":
                logits = np.log(
                    np.clip(model.predict_proba(features.reshape(1, -1))[0], 1e-7, 1)
                )
            else:
                import torch

                normalized = (features - np.array(meta["mean"], np.float32)) / np.array(
                    meta["scale"], np.float32
                )
                with torch.no_grad():
                    logits = model(
                        torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
                    ).numpy()[0]
            probabilities = softmax(logits / meta["temperature"])
            best = int(probabilities.argmax())
            ordered = np.sort(probabilities)
            margin = float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else 0.0
            return meta["classes"][best], float(probabilities[best]), margin
