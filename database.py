import sqlite3
from datetime import datetime
from config import DB_PATH, DEFAULT_SIGNS_CATALOG


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT UNIQUE NOT NULL,
            category TEXT NOT NULL,
            is_dynamic INTEGER NOT NULL DEFAULT 0,
            two_hands INTEGER NOT NULL DEFAULT 0,
            description TEXT,
            samples_count INTEGER NOT NULL DEFAULT 0,
            precision_score REAL,
            recall_score REAL,
            f1_score REAL,
            last_trained_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_summary (
            model_type TEXT PRIMARY KEY,
            global_accuracy REAL,
            total_samples INTEGER,
            trained_classes INTEGER,
            updated_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recognition_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            is_dynamic INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    # Popula o catálogo oficial se ainda não existir
    for item in DEFAULT_SIGNS_CATALOG:
        cursor.execute("""
            INSERT OR IGNORE INTO signs (label, category, is_dynamic, two_hands, description)
            VALUES (?, ?, ?, ?, ?)
        """, (item["label"], item["category"], item["is_dynamic"], item["two_hands"], item["description"]))

    conn.commit()
    conn.close()


def get_all_signs():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM signs ORDER BY category ASC, label ASC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_custom_sign(label: str, category: str, is_dynamic: int, two_hands: int, description: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO signs (label, category, is_dynamic, two_hands, description, samples_count)
        VALUES (
            ?, ?, ?, ?, ?,
            COALESCE((SELECT samples_count FROM signs WHERE label = ?), 0)
        )
    """, (label.strip(), category, int(bool(is_dynamic)), int(bool(two_hands)), description.strip(), label.strip()))
    conn.commit()
    conn.close()


def sync_sample_counts(counts_dict: dict):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE signs SET samples_count = 0")
    for label, count in counts_dict.items():
        cursor.execute("UPDATE signs SET samples_count = ? WHERE label = ?", (int(count), label))
    conn.commit()
    conn.close()


def update_training_metrics(model_type: str, global_accuracy: float, total_samples: int, class_metrics: dict):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO model_summary (model_type, global_accuracy, total_samples, trained_classes, updated_at)
        VALUES (?, ?, ?, ?, ?)
    """, (model_type, float(global_accuracy), int(total_samples), len(class_metrics), now))

    for label, m in class_metrics.items():
        cursor.execute("""
            UPDATE signs
            SET precision_score = ?, recall_score = ?, f1_score = ?, last_trained_at = ?
            WHERE label = ?
        """, (float(m["precision"]), float(m["recall"]), float(m["f1-score"]), now, label))

    conn.commit()
    conn.close()


def get_model_summaries():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM model_summary").fetchall()
    conn.close()
    return {r["model_type"]: dict(r) for r in rows}


def log_recognition(label: str, confidence: float, is_dynamic: int):
    now = datetime.now().strftime("%H:%M:%S")
    conn = get_connection()
    conn.execute("""
        INSERT INTO recognition_history (label, confidence, is_dynamic, created_at)
        VALUES (?, ?, ?, ?)
    """, (label, float(confidence), int(is_dynamic), now))
    conn.commit()
    conn.close()