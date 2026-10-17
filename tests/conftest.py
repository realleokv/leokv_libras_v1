import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import create_app
from config import TOTAL_FEATURES_DIM

TOKEN = "test-token-not-for-production-1234567890"


class FakeExtractor:
    instances = []

    def __init__(self):
        self.reset_count = 0
        self.motion_reset_count = 0
        self.closed = False
        self.__class__.instances.append(self)

    def reset(self):
        self.reset_count += 1

    def reset_motion(self):
        self.motion_reset_count += 1

    def close(self):
        self.closed = True

    def process_bgr_frame(self, frame, timestamp):
        x = np.zeros(TOTAL_FEATURES_DIM, np.float32)
        x[73] = 1
        x[155] = 1
        return {
            "features": x,
            "hands_detected": 1,
            "body_detected": True,
            "velocity": 0.0,
            "landmarks": [],
            "tracking_signature": ("Right",),
        }


@pytest.fixture
def app(tmp_path):
    FakeExtractor.instances = []
    return create_app(
        {
            "TESTING": True,
            "DATA_DIR": str(tmp_path),
            "ADMIN_TOKEN": TOKEN,
            "SECRET_KEY": "s" * 64,
        },
        FakeExtractor,
    )


@pytest.fixture
def client(app):
    c = app.test_client()
    c.csrf = c.get("/api/session").json["csrf"]
    return c


@pytest.fixture
def admin(client):
    assert (
        client.post(
            "/api/login", json={"token": TOKEN}, headers={"X-CSRF-Token": client.csrf}
        ).status_code
        == 200
    )
    return client


@pytest.fixture
def jpeg():
    b = io.BytesIO()
    Image.new("RGB", (64, 48)).save(b, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def sock(app, client):
    return app.extensions["libras_socket"].test_client(
        app, flask_test_client=client, auth={"csrf": client.csrf}
    )
