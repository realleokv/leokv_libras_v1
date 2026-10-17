"""Persistência transacional de gravações reais e metadados de avaliação."""

import io
import json
import sqlite3
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from config import (
    DB_PATH,
    DEFAULT_SIGNS_CATALOG,
    FEATURE_VERSION,
    BACKGROUND_LABEL,
    HISTORY_LIMIT,
)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def text_field(value, name, maximum=120, required=True):
    if not isinstance(value, str):
        raise ValueError(f"{name}: use texto.")
    value = unicodedata.normalize("NFC", value).strip()
    if (
        (required and not value)
        or len(value) > maximum
        or any(ord(c) < 32 for c in value)
    ):
        raise ValueError(
            f"{name}: preencha até {maximum} caracteres, sem caracteres de controle."
        )
    return value


def boolean_field(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name}: use verdadeiro ou falso.")
    return value


class Database:
    def __init__(self, path=DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def init_db(self):
        with self.connect() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS signs (
                    id INTEGER PRIMARY KEY, label TEXT NOT NULL,
                    canonical TEXT NOT NULL UNIQUE, category TEXT NOT NULL,
                    is_dynamic INTEGER NOT NULL, two_hands INTEGER NOT NULL,
                    requires_body INTEGER NOT NULL DEFAULT 0,
                    description TEXT NOT NULL DEFAULT '', reference_url TEXT NOT NULL DEFAULT '',
                    reviewer TEXT NOT NULL DEFAULT '', reviewed INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS recordings (
                    id TEXT PRIMARY KEY, label TEXT NOT NULL, kind TEXT NOT NULL,
                    signer TEXT NOT NULL, session TEXT NOT NULL, origin TEXT NOT NULL,
                    parent_id TEXT, feature_version INTEGER NOT NULL,
                    features BLOB NOT NULL, timestamps BLOB NOT NULL, created_at TEXT NOT NULL,
                    FOREIGN KEY(label) REFERENCES signs(canonical)
                );
                CREATE TABLE IF NOT EXISTS revisions (kind TEXT PRIMARY KEY, value INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS model_summary (kind TEXT PRIMARY KEY, metadata TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS recognition_history (
                    id INTEGER PRIMARY KEY, session TEXT NOT NULL, label TEXT NOT NULL,
                    score REAL NOT NULL, created_at TEXT NOT NULL
                );
            """
            )
            for kind in ("static", "dynamic"):
                c.execute("INSERT OR IGNORE INTO revisions VALUES (?, 0)", (kind,))
            for s in DEFAULT_SIGNS_CATALOG:
                c.execute(
                    """INSERT OR IGNORE INTO signs
                    (label,canonical,category,is_dynamic,two_hands,requires_body,description)
                    VALUES (?,?,?,?,?,?,?)""",
                    (
                        s["label"],
                        s["label"].casefold(),
                        s["category"],
                        s["is_dynamic"],
                        s["two_hands"],
                        int(s["category"] == "palavra" and s["label"] != "Casa"),
                        s["description"],
                    ),
                )
            c.execute(
                """INSERT OR IGNORE INTO signs
                (label,canonical,category,is_dynamic,two_hands,description) VALUES (?,?,?,0,0,?)""",
                (
                    BACKGROUND_LABEL,
                    BACKGROUND_LABEL.casefold(),
                    "controle",
                    "Nenhum sinal: mãos em repouso, movimentos comuns e transições. Colete nos dois modos.",
                ),
            )

    def get_signs(self):
        with self.connect() as c:
            rows = c.execute(
                """SELECT s.*, COUNT(r.id) recordings_count,
                COUNT(DISTINCT r.signer) signers_count
                FROM signs s LEFT JOIN recordings r ON s.canonical=r.label
                GROUP BY s.id ORDER BY s.category,s.label"""
            ).fetchall()
        return [dict(row) for row in rows]

    def get_sign(self, label):
        key = text_field(label, "Sinal").casefold()
        with self.connect() as c:
            row = c.execute("SELECT * FROM signs WHERE canonical=?", (key,)).fetchone()
        if row is None:
            raise ValueError("Sinal não cadastrado.")
        return dict(row)

    def upsert_sign(self, data):
        label = text_field(data.get("label"), "Nome", 60)
        if len(label) == 1 and label.isalpha():
            label = label.upper()
        if label.casefold() == BACKGROUND_LABEL.casefold():
            raise ValueError("A classe de controle é reservada.")
        category = data.get("category")
        if category not in ("alfabeto", "palavra"):
            raise ValueError("Categoria inválida.")
        dynamic = boolean_field(data.get("is_dynamic"), "Movimento")
        two = boolean_field(data.get("two_hands"), "Duas mãos")
        body = boolean_field(data.get("requires_body", False), "Corpo")
        reviewed = boolean_field(data.get("reviewed", False), "Revisado")
        desc = text_field(data.get("description", ""), "Descrição", 1000, False)
        url = text_field(data.get("reference_url", ""), "Referência", 500, False)
        if url:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            if parsed.scheme != "https" or not parsed.hostname or parsed.username:
                raise ValueError("Use um link HTTPS de referência ou deixe vazio.")
        reviewer = text_field(data.get("reviewer", ""), "Revisor", 100, reviewed)
        if reviewed and not url:
            raise ValueError("Registre a referência da revisão linguística.")
        with self.connect() as c:
            old = c.execute(
                "SELECT * FROM signs WHERE canonical=?", (label.casefold(),)
            ).fetchone()
            count = c.execute(
                "SELECT COUNT(*) FROM recordings WHERE label=?", (label.casefold(),)
            ).fetchone()[0]
            if (
                old
                and count
                and (
                    old["is_dynamic"] != dynamic
                    or old["two_hands"] != two
                    or old["requires_body"] != body
                )
            ):
                raise ValueError(
                    "Apague ou exporte as gravações antes de mudar os requisitos do sinal."
                )
            c.execute(
                """INSERT INTO signs
                (label,canonical,category,is_dynamic,two_hands,requires_body,description,reference_url,reviewer,reviewed)
                VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(canonical) DO UPDATE SET
                category=excluded.category,is_dynamic=excluded.is_dynamic,two_hands=excluded.two_hands,
                requires_body=excluded.requires_body,description=excluded.description,
                reference_url=excluded.reference_url,reviewer=excluded.reviewer,reviewed=excluded.reviewed""",
                (
                    label,
                    label.casefold(),
                    category,
                    dynamic,
                    two,
                    body,
                    desc,
                    url,
                    reviewer,
                    reviewed,
                ),
            )

    @staticmethod
    def pack(array):
        b = io.BytesIO()
        np.save(b, np.asarray(array), allow_pickle=False)
        return b.getvalue()

    @staticmethod
    def unpack(blob):
        return np.load(io.BytesIO(blob), allow_pickle=False)

    def save_recording(
        self, recording_id, label, kind, signer, session, features, timestamps
    ):
        # Só frames extraídos pelo servidor chegam aqui; não há upload público de features.
        sign = self.get_sign(label)
        with self.connect() as c:
            c.execute(
                "INSERT INTO recordings VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    recording_id,
                    sign["canonical"],
                    kind,
                    signer,
                    session,
                    "real",
                    None,
                    FEATURE_VERSION,
                    self.pack(features),
                    self.pack(timestamps),
                    utc_now(),
                ),
            )
            c.execute("UPDATE revisions SET value=value+1 WHERE kind=?", (kind,))

    def delete_samples(self, label):
        sign = self.get_sign(label)
        with self.connect() as c:
            kinds = c.execute(
                "SELECT DISTINCT kind FROM recordings WHERE label=?",
                (sign["canonical"],),
            ).fetchall()
            c.execute("DELETE FROM recordings WHERE label=?", (sign["canonical"],))
            for row in kinds:
                c.execute("UPDATE revisions SET value=value+1 WHERE kind=?", (row[0],))
        return [row[0] for row in kinds]

    def revision(self, kind):
        with self.connect() as c:
            return c.execute(
                "SELECT value FROM revisions WHERE kind=?", (kind,)
            ).fetchone()[0]

    def recordings(self, kind):
        with self.connect() as c:
            rows = c.execute(
                """SELECT r.*,s.label display_label FROM recordings r
                JOIN signs s ON r.label=s.canonical WHERE r.kind=?
                AND origin='real' AND feature_version=? ORDER BY created_at,id""",
                (kind, FEATURE_VERSION),
            ).fetchall()
        return [
            dict(
                r,
                features=self.unpack(r["features"]),
                timestamps=self.unpack(r["timestamps"]),
            )
            for r in rows
        ]

    def set_summary(self, kind, metadata):
        with self.connect() as c:
            c.execute(
                "INSERT INTO model_summary VALUES (?,?) ON CONFLICT(kind) DO UPDATE SET metadata=excluded.metadata",
                (kind, json.dumps(metadata, ensure_ascii=False)),
            )

    def summaries(self):
        with self.connect() as c:
            rows = c.execute("SELECT * FROM model_summary").fetchall()
        return {r["kind"]: json.loads(r["metadata"]) for r in rows}

    def log_recognition(self, session_id, label, score):
        with self.connect() as c:
            c.execute(
                "INSERT INTO recognition_history(session,label,score,created_at) VALUES (?,?,?,?)",
                (session_id, label, float(score), utc_now()),
            )
            c.execute(
                """DELETE FROM recognition_history WHERE id NOT IN
                (SELECT id FROM recognition_history ORDER BY id DESC LIMIT ?)""",
                (HISTORY_LIMIT,),
            )

    def history(self):
        with self.connect() as c:
            return [
                dict(r)
                for r in c.execute(
                    "SELECT * FROM recognition_history ORDER BY id DESC LIMIT ?",
                    (HISTORY_LIMIT,),
                )
            ]

    def clear_history(self):
        with self.connect() as c:
            c.execute("DELETE FROM recognition_history")
