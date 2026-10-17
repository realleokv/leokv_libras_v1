import base64
import io
import time
import pytest
from PIL import Image
from app import decode_base64_frame
from conftest import sock, FakeExtractor, TOKEN


def post(client, path, data):
    return client.post(path, json=data, headers={"X-CSRF-Token": client.csrf})


def test_no_auth_no_mutation(client):
    for path in (
        "/api/signs",
        "/api/samples/delete",
        "/api/train",
        "/api/history/clear",
    ):
        assert post(client, path, {}).status_code == 403


def test_csrf_origin_and_malformed_json(admin):
    assert admin.post("/api/history/clear", json={}).status_code == 403
    assert (
        admin.post(
            "/api/history/clear", json={}, headers={"X-CSRF-Token": "inválido"}
        ).status_code
        == 403
    )
    assert (
        admin.post(
            "/api/history/clear",
            json={},
            headers={"X-CSRF-Token": admin.csrf, "Origin": "https://untrusted.example"},
        ).status_code
        == 403
    )
    assert post(admin, "/api/signs", []).status_code == 400
    assert (
        admin.get("/")
        .headers["Content-Security-Policy"]
        .startswith("default-src 'self'")
    )


def test_login_rate_limit(client):
    for _ in range(5):
        assert post(client, "/api/login", {"token": "bad"}).status_code == 403
    assert post(client, "/api/login", {"token": TOKEN}).status_code == 429


def test_strict_boolean_and_preserved_identity(admin, app):
    data = {
        "label": "Sinal d'água <texto>",
        "category": "palavra",
        "is_dynamic": False,
        "two_hands": True,
        "description": "Teste",
    }
    assert post(admin, "/api/signs", dict(data, is_dynamic="false")).status_code == 400
    assert post(admin, "/api/signs", data).status_code == 200
    db = app.extensions["libras"].db
    before = db.get_sign(data["label"])
    assert (
        post(admin, "/api/signs", dict(data, description="Nova descrição")).status_code
        == 200
    )
    assert db.get_sign(data["label"])["id"] == before["id"]


def test_j_has_no_stale_metrics(client):
    data = client.get("/api/signs").json
    j = next(s for s in data["signs"] if s["label"] == "J")
    assert j["recordings_count"] == 0
    assert data["models"]["dynamic"]["metrics"] is None
    assert not data["models"]["dynamic"]["ready"]


def test_websocket_auth_and_isolation(app, client, jpeg):
    sio = app.extensions["libras_socket"]
    bad = sio.test_client(app, flask_test_client=client, auth={"csrf": "wrong"})
    assert not bad.is_connected()
    for invalid in ("inválido", None, [], 123):
        rejected = sio.test_client(
            app, flask_test_client=client, auth={"csrf": invalid}
        )
        assert not rejected.is_connected()
    a = sock(app, client)
    second = app.test_client()
    second.csrf = second.get("/api/session").json["csrf"]
    b = sock(app, second)
    assert a.emit(
        "configure_threshold", {"static": 0.91, "dynamic": 0.93}, callback=True
    )["ok"]
    states = list(app.extensions["libras"].clients.values())
    assert states[0].stabilizer.thresholds["static"] == 0.91
    assert states[1].stabilizer.thresholds["static"] == 0.8
    for connection in (a, b):
        response = connection.emit(
            "process_frame",
            {"image": jpeg, "timestamp": 1, "generation": "g", "frame_id": 1},
            callback=True,
        )
        assert response["ok"]
    assert len(FakeExtractor.instances) == 2
    a.disconnect()
    assert FakeExtractor.instances[0].closed
    assert not FakeExtractor.instances[1].closed
    b.disconnect()


def test_every_invalid_frame_gets_response_and_next_can_work(app, client, jpeg):
    connection = sock(app, client)
    runtime = app.extensions["libras"]
    clock = [10.0]
    runtime.clock = lambda: clock[0]
    assert not connection.emit(
        "process_frame",
        {"image": "bad", "timestamp": 1, "generation": "x"},
        callback=True,
    )["ok"]
    clock[0] += 0.1
    response = connection.emit(
        "process_frame",
        {"image": jpeg, "timestamp": 1.1, "generation": "x"},
        callback=True,
    )
    assert response["ok"]
    connection.disconnect()


def test_server_capture_no_client_features_and_logout_revokes(app, admin, jpeg):
    connection = sock(app, admin)
    runtime = app.extensions["libras"]
    now = [10.0]
    runtime.clock = lambda: now[0]

    # Login usado um relógio real; continua válido ao substituir por relógio de teste.
    def frame(t):
        now[0] = 10 + t
        return connection.emit(
            "process_frame",
            {"image": jpeg, "timestamp": t, "generation": "g"},
            callback=True,
        )

    assert frame(1)["ok"]
    spec = {
        "label": "A",
        "kind": "static",
        "signer": "p01",
        "session": "s01",
        "count": 6,
        "generation": "g",
    }
    assert connection.emit("start_capture", spec, callback=True)["ok"]
    for i in range(6):
        response = frame(1.25 + i * 0.25)
    assert response["capture"]["saved"]
    records = runtime.db.recordings("static")
    assert len(records) == 1 and records[0]["origin"] == "real"
    assert records[0]["features"].shape == (6, 173)
    assert post(admin, "/api/logout", {}).status_code == 200
    assert not connection.emit("start_capture", spec, callback=True)["ok"]
    connection.disconnect()


def test_training_job_returns_actionable_validation_error(app, admin):
    response = post(admin, "/api/train", {"kind": "static"})
    assert response.status_code == 202
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        status = admin.get("/api/train/status").json
        if status["status"] != "running":
            break
        time.sleep(0.01)
    assert status["status"] == "done"
    assert not status["results"]["static"]["trained"]
    assert "dois sinais" in status["results"]["static"]["error"]


def test_frame_gap_resets_motion_without_restarting_detector(app, client, jpeg):
    connection = sock(app, client)
    runtime = app.extensions["libras"]
    now = [10.0]
    runtime.clock = lambda: now[0]
    for timestamp in (1.0, 2.0):
        now[0] += 1
        response = connection.emit(
            "process_frame",
            {"image": jpeg, "timestamp": timestamp, "generation": "g"},
            callback=True,
        )
        assert response["ok"]
    extractor = FakeExtractor.instances[0]
    assert extractor.motion_reset_count == 1
    assert extractor.reset_count == 0
    connection.disconnect()


def test_deleting_samples_cancels_capture_and_prevents_repopulation(app, admin, jpeg):
    connection = sock(app, admin)
    runtime = app.extensions["libras"]
    now = [10.0]
    runtime.clock = lambda: now[0]

    def frame(timestamp):
        now[0] += 0.25
        return connection.emit(
            "process_frame",
            {"image": jpeg, "timestamp": timestamp, "generation": "g"},
            callback=True,
        )

    assert frame(1)["ok"]
    assert connection.emit(
        "start_capture",
        {
            "label": "A",
            "kind": "static",
            "signer": "p01",
            "session": "s01",
            "count": 6,
            "generation": "g",
        },
        callback=True,
    )["ok"]
    assert frame(1.25)["capture"]["active"]
    assert post(admin, "/api/samples/delete", {"label": "A"}).status_code == 200
    for timestamp in (1.5, 1.75, 2.0, 2.25, 2.5, 2.75):
        response = frame(timestamp)
        assert response["ok"] and not response["capture"]["active"]
    assert runtime.db.recordings("static") == []
    connection.disconnect()


def test_mutation_is_blocked_while_training(app, admin):
    runtime = app.extensions["libras"]
    runtime.edit_lock.acquire()
    try:
        assert post(admin, "/api/samples/delete", {"label": "A"}).status_code == 409
    finally:
        runtime.edit_lock.release()


def test_bad_image_and_resolution_rejected(jpeg):
    assert decode_base64_frame(jpeg).shape == (48, 64, 3)
    with pytest.raises(ValueError):
        decode_base64_frame("data:image/jpeg;base64,%%%%")
    b = io.BytesIO()
    Image.new("RGB", (2000, 2000)).save(b, format="JPEG")
    with pytest.raises(ValueError):
        decode_base64_frame(
            "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
        )
