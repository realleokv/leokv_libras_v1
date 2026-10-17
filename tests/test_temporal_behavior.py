import numpy as np
import pytest
from stabilizer import TemporalStabilizer
from capture import Capture
from config import TOTAL_FEATURES_DIM


class Engine:
    def __init__(self, label="A", dynamic=False):
        self.label = label
        self.dynamic = dynamic
        self.calls = 0

    def available(self, kind):
        return (kind == "dynamic") == self.dynamic

    def predict(self, kind, x):
        self.calls += 1
        return self.label, 0.97, 0.9


SIGNS = {
    "a": {"two_hands": False, "requires_body": False},
    "j": {"two_hands": False, "requires_body": False},
}


def extracted(hands=1, velocity=0, body=True):
    x = np.zeros(TOTAL_FEATURES_DIM, np.float32)
    x[73] = int(hands > 0)
    x[155] = int(body)
    return dict(
        features=x,
        hands_detected=hands,
        velocity=velocity,
        body_detected=body,
        tracking_signature=("Right",) if hands else (),
    )


def test_held_letter_is_emitted_only_once_and_neutral_allows_repeat():
    stabilizer = TemporalStabilizer()
    engine = Engine()
    emitted = []
    for t in np.arange(0, 5, 0.1):
        r = stabilizer.process_step(extracted(), float(t), engine, SIGNS)
        if r["emitted"]:
            emitted.append(r["emitted"])
    assert emitted == ["A"]
    for t in np.arange(5, 5.7, 0.1):
        stabilizer.process_step(extracted(hands=0), float(t), engine, SIGNS)
    for t in np.arange(5.7, 6.5, 0.1):
        r = stabilizer.process_step(extracted(), float(t), engine, SIGNS)
        if r["emitted"]:
            emitted.append(r["emitted"])
    assert emitted == ["A", "A"]


def test_movement_clears_old_static_votes():
    s = TemporalStabilizer()
    engine = Engine()
    for t in (0, 0.1, 0.2):
        s.process_step(extracted(), t, engine, SIGNS)
    assert len(s.static_votes) == 3
    s.process_step(extracted(velocity=1), 0.3, engine, SIGNS)
    assert not s.static_votes


def test_dynamic_gesture_is_evaluated_after_motion_stops():
    s = TemporalStabilizer()
    engine = Engine("J", True)
    emitted = []
    for t in np.arange(0, 2.21, 0.1):
        r = s.process_step(
            extracted(velocity=0.4 if t < 1 else 0), float(t), engine, SIGNS
        )
        if r["emitted"]:
            emitted.append(r["emitted"])
    assert engine.calls >= 2 and emitted == ["J"]


def test_capture_rejects_missing_second_hand_and_gaps():
    sign = {"label": "Casa", "is_dynamic": 0, "two_hands": 1, "requires_body": 0}
    data = {"kind": "static", "signer": "p01", "session": "s01", "count": 6}
    c = Capture(sign, data, 0)
    with pytest.raises(ValueError):
        c.add(extracted(hands=1), 0, 0)
    c = Capture(dict(sign, two_hands=0), data, 0)
    c.add(extracted(), 0, 0)
    with pytest.raises(ValueError):
        c.add(extracted(), 1, 1)


def test_capture_rejects_missing_body():
    c = Capture(
        {"label": "Eu", "is_dynamic": 0, "two_hands": 0, "requires_body": 1},
        {"kind": "static", "signer": "p", "session": "s", "count": 6},
        0,
    )
    with pytest.raises(ValueError):
        c.add(extracted(body=False), 0, 0)
