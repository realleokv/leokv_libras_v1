import uuid
import numpy as np
import pytest
from database import Database
from ml_engine import LibrasMLEngine, grouped_partitions
from config import TOTAL_FEATURES_DIM, BACKGROUND_LABEL


def fixtures(db, kind="static"):
    # Coordenadas artificiais usadas somente para testar o SOFTWARE, nunca sua precisão real.
    rng = np.random.default_rng(8)
    labels = (
        ["A", "B", BACKGROUND_LABEL]
        if kind == "static"
        else ["H", "J", BACKGROUND_LABEL]
    )
    for person in range(3):
        for number, label in enumerate(labels):
            for take in range(2):
                times = np.linspace(0, 2, 21)
                x = rng.normal(number, 0.02, (21, TOTAL_FEATURES_DIM)).astype(
                    np.float32
                )
                db.save_recording(
                    uuid.uuid4().hex, label, kind, f"p{person}", f"s{take}", x, times
                )
    return db.recordings(kind)


def test_people_do_not_cross_partitions(tmp_path):
    records = fixtures(Database(tmp_path / "db"))
    parts = grouped_partitions(records)
    groups = [{records[i]["signer"] for i in p} for p in parts]
    assert (
        not groups[0] & groups[1]
        and not groups[0] & groups[2]
        and not groups[1] & groups[2]
    )


def test_insufficient_samples_returns_clear_error(tmp_path):
    records = fixtures(Database(tmp_path / "db"))
    with pytest.raises(ValueError, match="6 gravações"):
        grouped_partitions(records[:-1])


def test_static_model_roundtrip_and_invalidation(tmp_path):
    db = Database(tmp_path / "db")
    fixtures(db)
    engine = LibrasMLEngine(db, tmp_path / "models")
    meta = engine.train("static", lambda m: None)
    assert meta["test_recordings"] == 6 and meta["split_people_counts"] == [1, 1, 1]
    restored = LibrasMLEngine(db, tmp_path / "models")
    assert restored.available("static"), restored.errors
    probe = db.recordings("static")[0]["features"][0]
    assert engine.predict("static", probe) == restored.predict("static", probe)
    kinds = db.delete_samples("A")
    for kind in kinds:
        restored.invalidate(kind)
    assert not restored.status()["static"]["ready"]
    assert restored.status()["static"]["metrics"] is None
    fresh = LibrasMLEngine(db, tmp_path / "models")
    assert not fresh.available("static")


def test_dynamic_model_roundtrip_safe_weights(tmp_path):
    db = Database(tmp_path / "db")
    fixtures(db, "dynamic")
    engine = LibrasMLEngine(db, tmp_path / "models")
    meta = engine.train("dynamic", lambda m: None, epochs=2)
    assert "J" in meta["classes"]
    restored = LibrasMLEngine(db, tmp_path / "models")
    assert restored.available("dynamic"), restored.errors


def test_existing_recording_blocks_incompatible_sign_edit(tmp_path):
    db = Database(tmp_path / "db")
    fixtures(db)
    with pytest.raises(ValueError, match="requisitos"):
        db.upsert_sign(
            {
                "label": "A",
                "category": "alfabeto",
                "is_dynamic": True,
                "two_hands": False,
            }
        )


def test_history_is_bounded_with_full_date(tmp_path):
    db = Database(tmp_path / "db")
    for i in range(1004):
        db.log_recognition("test", "A", 90)
    history = db.history()
    assert (
        len(history) == 1000
        and "T" in history[0]["created_at"]
        and "+00:00" in history[0]["created_at"]
    )
