import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from config import (
    STATIC_DATA_PATH,
    DYNAMIC_DATA_PATH,
    STATIC_MODEL_PATH,
    DYNAMIC_MODEL_PATH,
    TOTAL_FEATURES_DIM,
    SEQ_LENGTH
)
from database import sync_sample_counts, update_training_metrics


class LibrasLSTM(nn.Module):
    """Rede Neural Recorrente LSTM para sinais dinâmicos de Libras."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        logits = self.fc(last_step)
        return logits


class LibrasMLEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.static_model = None
        self.static_classes = []

        self.dynamic_model = None
        self.dynamic_classes = []

        self.load_models()
        self.refresh_database_counts()

    # -------------------- Gerenciamento de Amostras -------------------- #
    @staticmethod
    def _load_npz(path: str, empty_shape: tuple):
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            return data["X"], data["y"]
        return np.empty(empty_shape, dtype=np.float32), np.empty((0,), dtype=str)

    def add_static_samples(self, features_list: list, label: str) -> int:
        X_new = np.array(features_list, dtype=np.float32)
        y_new = np.array([label] * len(features_list), dtype=str)

        X_old, y_old = self._load_npz(STATIC_DATA_PATH, (0, TOTAL_FEATURES_DIM))
        X_all = np.vstack([X_old, X_new]) if len(X_old) > 0 else X_new
        y_all = np.concatenate([y_old, y_new]) if len(y_old) > 0 else y_new

        np.savez_compressed(STATIC_DATA_PATH, X=X_all, y=y_all)
        self.refresh_database_counts()
        return int(np.sum(y_all == label))

    def add_dynamic_sequence(self, sequence: np.ndarray, label: str) -> int:
        seq_arr = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)  # (1, SEQ_LENGTH, 146)
        y_new = np.array([label], dtype=str)

        X_old, y_old = self._load_npz(DYNAMIC_DATA_PATH, (0, SEQ_LENGTH, TOTAL_FEATURES_DIM))
        X_all = np.vstack([X_old, seq_arr]) if len(X_old) > 0 else seq_arr
        y_all = np.concatenate([y_old, y_new]) if len(y_old) > 0 else y_new

        np.savez_compressed(DYNAMIC_DATA_PATH, X=X_all, y=y_all)
        self.refresh_database_counts()
        return int(np.sum(y_all == label))

    def delete_samples_for_label(self, label: str):
        # Remove de estáticos
        X_s, y_s = self._load_npz(STATIC_DATA_PATH, (0, TOTAL_FEATURES_DIM))
        if len(y_s) > 0:
            mask = y_s != label
            np.savez_compressed(STATIC_DATA_PATH, X=X_s[mask], y=y_s[mask])

        # Remove de dinâmicos
        X_d, y_d = self._load_npz(DYNAMIC_DATA_PATH, (0, SEQ_LENGTH, TOTAL_FEATURES_DIM))
        if len(y_d) > 0:
            mask = y_d != label
            np.savez_compressed(DYNAMIC_DATA_PATH, X=X_d[mask], y=y_d[mask])

        self.refresh_database_counts()

    def refresh_database_counts(self):
        counts = {}
        _, y_s = self._load_npz(STATIC_DATA_PATH, (0, TOTAL_FEATURES_DIM))
        _, y_d = self._load_npz(DYNAMIC_DATA_PATH, (0, SEQ_LENGTH, TOTAL_FEATURES_DIM))

        for lbl in y_s:
            counts[str(lbl)] = counts.get(str(lbl), 0) + 1
        for lbl in y_d:
            counts[str(lbl)] = counts.get(str(lbl), 0) + 1

        sync_sample_counts(counts)

    # -------------------- Carregamento de Modelos -------------------- #
    def load_models(self):
        if os.path.exists(STATIC_MODEL_PATH):
            try:
                bundle = joblib.load(STATIC_MODEL_PATH)
                self.static_model = bundle["model"]
                self.static_classes = list(bundle["classes"])
            except Exception as e:
                print(f"[AVISO] Falha ao carregar modelo estático: {e}")

        if os.path.exists(DYNAMIC_MODEL_PATH):
            try:
                checkpoint = torch.load(DYNAMIC_MODEL_PATH, map_location=self.device, weights_only=False)
                classes = checkpoint["classes"]
                model = LibrasLSTM(
                    input_size=TOTAL_FEATURES_DIM,
                    hidden_size=128,
                    num_layers=2,
                    num_classes=len(classes)
                ).to(self.device)
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()
                self.dynamic_model = model
                self.dynamic_classes = list(classes)
            except Exception as e:
                print(f"[AVISO] Falha ao carregar modelo dinâmico: {e}")

    # -------------------- Treinamento Estático (Scikit-Learn) -------------------- #
    def train_static_model(self) -> dict:
        X, y = self._load_npz(STATIC_DATA_PATH, (0, TOTAL_FEATURES_DIM))
        unique_classes, counts = np.unique(y, return_counts=True)

        if len(unique_classes) < 2:
            return {
                "trained": False,
                "message": "São necessárias amostras de pelo menos 2 sinais estáticos diferentes para treinar o classificador Scikit-learn."
            }

        if np.min(counts) < 4:
            return {
                "trained": False,
                "message": f"Cada sinal estático precisa de pelo menos 4 amostras (mínimo atual: {int(np.min(counts))})."
            }

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Treina com 100% dos dados após avaliar na partição de teste para máxima performance em produção
        clf.fit(X, y)
        joblib.dump({"model": clf, "classes": list(clf.classes_)}, STATIC_MODEL_PATH)

        self.static_model = clf
        self.static_classes = list(clf.classes_)

        class_metrics = {
            cls: report[cls]
            for cls in unique_classes
            if cls in report
        }
        update_training_metrics("static_rf", acc, len(y), class_metrics)

        return {
            "trained": True,
            "accuracy": round(acc * 100, 2),
            "classes": list(unique_classes),
            "total_samples": len(y),
            "message": f"Modelo estático (Random Forest) treinado com {len(unique_classes)} classes e {round(acc*100, 1)}% de acurácia real em validação."
        }

    # -------------------- Treinamento Dinâmico (PyTorch LSTM) -------------------- #
    def train_dynamic_model(self, epochs: int = 45) -> dict:
        X, y = self._load_npz(DYNAMIC_DATA_PATH, (0, SEQ_LENGTH, TOTAL_FEATURES_DIM))
        unique_classes, counts = np.unique(y, return_counts=True)

        if len(unique_classes) < 2:
            return {
                "trained": False,
                "message": "São necessárias sequências de pelo menos 2 sinais dinâmicos diferentes para treinar a LSTM."
            }

        if np.min(counts) < 3:
            return {
                "trained": False,
                "message": f"Cada sinal dinâmico precisa de pelo menos 3 sequências gravadas (mínimo atual: {int(np.min(counts))})."
            }

        classes = sorted(list(unique_classes))
        label_to_idx = {c: i for i, c in enumerate(classes)}
        y_idx = np.array([label_to_idx[lbl] for lbl in y], dtype=np.int64)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_idx, test_size=0.25, random_state=42, stratify=y_idx
        )

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        train_loader = DataLoader(train_ds, batch_size=min(16, len(train_ds)), shuffle=True)

        model = LibrasLSTM(
            input_size=TOTAL_FEATURES_DIM,
            hidden_size=128,
            num_layers=2,
            num_classes=len(classes)
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

        model.train()
        for _ in range(epochs):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            test_x = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            logits = model(test_x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        y_test_labels = [classes[i] for i in y_test]
        y_pred_labels = [classes[i] for i in preds]

        acc = float(accuracy_score(y_test_labels, y_pred_labels))
        report = classification_report(y_test_labels, y_pred_labels, output_dict=True, zero_division=0)

        torch.save({
            "state_dict": model.state_dict(),
            "classes": classes
        }, DYNAMIC_MODEL_PATH)

        self.dynamic_model = model
        self.dynamic_classes = classes

        class_metrics = {
            cls: report[cls]
            for cls in classes
            if cls in report
        }
        update_training_metrics("dynamic_lstm", acc, len(y), class_metrics)

        return {
            "trained": True,
            "accuracy": round(acc * 100, 2),
            "classes": classes,
            "total_samples": len(y),
            "message": f"Modelo temporal (PyTorch LSTM) treinado com {len(classes)} classes e {round(acc*100, 1)}% de acurácia real em validação."
        }

    # -------------------- Inferência em Tempo Real -------------------- #
    def predict_static(self, features: np.ndarray) -> tuple:
        if self.static_model is None or len(self.static_classes) == 0:
            return None, 0.0
        probs = self.static_model.predict_proba(features.reshape(1, -1))[0]
        best_idx = int(np.argmax(probs))
        return str(self.static_model.classes_[best_idx]), float(probs[best_idx])

    def predict_dynamic(self, sequence: np.ndarray) -> tuple:
        if self.dynamic_model is None or len(self.dynamic_classes) == 0:
            return None, 0.0
        with torch.no_grad():
            tensor_seq = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits = self.dynamic_model(tensor_seq)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            best_idx = int(np.argmax(probs))
            return str(self.dynamic_classes[best_idx]), float(probs[best_idx])