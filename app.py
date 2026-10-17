"""Servidor local do protótipo LibrasVision v2."""

import argparse
import base64
import binascii
import hmac
import io
import ipaddress
import logging
import math
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, session
from flask_socketio import SocketIO
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

from config import (
    BASE_DIR,
    DATA_DIR,
    FEATURE_VERSION,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS,
    MAX_SESSIONS,
    MAX_FRAME_GAP,
    SEQ_LENGTH,
)
from database import Database
from feature_extractor import HandFeatureExtractor
from ml_engine import LibrasMLEngine
from stabilizer import TemporalStabilizer, result
from capture import Capture


def decode_base64_frame(data_url):
    if not isinstance(data_url, str) or not data_url.startswith(
        "data:image/jpeg;base64,"
    ):
        raise ValueError("Envie um frame JPEG.")
    if len(data_url) > MAX_IMAGE_BYTES * 4 // 3 + 40:
        raise ValueError("Imagem muito grande.")
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            w, h = image.size
            if (
                image.format != "JPEG"
                or min(w, h) < 32
                or max(w, h) > 1280
                or w * h > MAX_IMAGE_PIXELS
            ):
                raise ValueError("Resolução inválida. Limite: 1280 × 960 pixels.")
            image.verify()
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Não foi possível decodificar o frame.")
        return frame
    except (
        binascii.Error,
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError("Imagem JPEG inválida.") from exc


class ClientState:
    def __init__(self, owner):
        self.owner = owner
        self.lock = threading.Lock()
        self.extractor = None
        self.stabilizer = TemporalStabilizer()
        self.capture = None
        self.generation = None
        self.last_client_time = None
        self.last_received = 0.0
        self.last_activity = time.monotonic()

    def reset(self):
        self.capture = None
        self.last_client_time = None
        self.stabilizer.reset()
        if self.extractor:
            self.extractor.reset()

    def close(self):
        if self.extractor:
            self.extractor.close()
            self.extractor = None


class Runtime:
    def __init__(self, database, engine, extractor_factory):
        self.db, self.engine, self.extractor_factory = (
            database,
            engine,
            extractor_factory,
        )
        self.clients = {}
        self.admins = {}
        self.clients_lock = threading.RLock()
        self.edit_lock = threading.Lock()
        self.job_lock = threading.Lock()
        self.job = {"status": "idle", "message": "Nenhum treinamento em execução."}
        self.limit_lock = threading.Lock()
        self.limits = defaultdict(deque)
        self.clock = time.monotonic
        self.signs = {s["canonical"]: s for s in database.get_signs()}
        self.camera_error = None

    def limited(self, key, limit, seconds):
        now = self.clock()
        with self.limit_lock:
            # Remove chaves antigas e limita memória mesmo com muitos endereços.
            for old in list(self.limits):
                if not self.limits[old] or now - self.limits[old][-1] > 3600:
                    del self.limits[old]
            if key not in self.limits and len(self.limits) >= 2048:
                return True
            q = self.limits[key]
            while q and now - q[0] >= seconds:
                q.popleft()
            if len(q) >= limit:
                return True
            q.append(now)
            return False

    def is_admin(self):
        with self.clients_lock:
            return self.admins.get(session.get("client_id"), 0) > self.clock()


def create_app(test_config=None, extractor_factory=HandFeatureExtractor):
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("LIBRAS_SECRET_KEY") or secrets.token_hex(32),
        ADMIN_TOKEN=os.environ.get("LIBRAS_ADMIN_TOKEN"),
        DATA_DIR=str(DATA_DIR),
        MAX_CONTENT_LENGTH=600_000,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict",
        SESSION_COOKIE_SECURE=False,
        LOG_HISTORY=os.environ.get("LIBRAS_LOG_HISTORY") == "1",
    )
    if test_config:
        app.config.update(test_config)
    for key in ("SECRET_KEY", "ADMIN_TOKEN"):
        value = app.config.get(key)
        if value is not None and (not isinstance(value, str) or len(value) < 32):
            raise ValueError(f"{key} precisa de pelo menos 32 caracteres.")
    origins = [
        v.strip().rstrip("/")
        for v in os.environ.get("LIBRAS_ALLOWED_ORIGINS", "").split(",")
        if v.strip()
    ]
    if "*" in origins:
        raise ValueError("Defina origens explícitas; o curinga não é permitido.")
    # None mantém a verificação padrão de mesma origem do Engine.IO.
    socketio = SocketIO(
        app,
        cors_allowed_origins=origins or None,
        async_mode="threading",
        max_http_buffer_size=600_000,
        manage_session=False,
        logger=False,
        engineio_logger=False,
    )
    database = Database(Path(app.config["DATA_DIR"]) / "libras_v2.db")
    runtime = Runtime(
        database,
        LibrasMLEngine(database, Path(app.config["DATA_DIR"]) / "models_v2"),
        extractor_factory,
    )
    app.extensions["libras"] = runtime
    app.extensions["libras_socket"] = socketio

    @app.before_request
    def protect_request():
        if request.path.startswith("/api/"):
            session.setdefault("csrf", secrets.token_urlsafe(32))
            session.setdefault("client_id", secrets.token_urlsafe(24))
        if request.method in ("POST", "DELETE", "PUT", "PATCH"):
            origin = request.headers.get("Origin")
            if origin and origin.rstrip("/") not in [
                request.host_url.rstrip("/"),
                *origins,
            ]:
                return jsonify(error="Origem não permitida."), 403
            if not hmac.compare_digest(
                request.headers.get("X-CSRF-Token", ""), session.get("csrf", "!")
            ):
                return jsonify(error="Sessão expirada. Recarregue a página."), 403

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Permissions-Policy"] = "camera=(self), microphone=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; media-src 'self' blob:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'"
        )
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(ValueError)
    def bad_value(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    @app.errorhandler(Exception)
    def unexpected_error(error):
        app.logger.exception("Falha interna")
        return jsonify(error="Falha interna. Consulte o terminal do servidor."), 500

    def payload():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ValueError("Envie um objeto JSON válido.")
        return data

    def admin_required():
        if not runtime.is_admin():
            from werkzeug.exceptions import Forbidden

            raise Forbidden("Entre em Administração para realizar esta ação.")

    def begin_edit():
        admin_required()
        if not runtime.edit_lock.acquire(blocking=False):
            from werkzeug.exceptions import Conflict

            raise Conflict("Treinamento em andamento. Aguarde antes de alterar dados.")

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/session")
    def api_session():
        return jsonify(
            csrf=session["csrf"],
            admin=runtime.is_admin(),
            admin_configured=bool(app.config["ADMIN_TOKEN"]),
        )

    @app.post("/api/login")
    def login():
        if runtime.limited(("login", request.remote_addr), 5, 60):
            return jsonify(error="Muitas tentativas. Aguarde um minuto."), 429
        token = payload().get("token")
        if (
            not isinstance(token, str)
            or len(token) > 256
            or not app.config["ADMIN_TOKEN"]
            or not hmac.compare_digest(
                token.encode("utf-8"), app.config["ADMIN_TOKEN"].encode("utf-8")
            )
        ):
            return (
                jsonify(error="Token administrativo inválido ou não configurado."),
                403,
            )
        with runtime.clients_lock:
            runtime.admins = {
                k: v for k, v in runtime.admins.items() if v > runtime.clock()
            }
            runtime.admins[session["client_id"]] = runtime.clock() + 3600
        return jsonify(ok=True)

    @app.post("/api/logout")
    def logout():
        with runtime.clients_lock:
            runtime.admins.pop(session.get("client_id"), None)
            affected = [
                s
                for s in runtime.clients.values()
                if s.owner == session.get("client_id")
            ]
        for state in affected:
            with state.lock:
                state.capture = None
        return jsonify(ok=True)

    @app.get("/api/health")
    def health():
        return jsonify(
            feature_version=FEATURE_VERSION,
            models=runtime.engine.status(),
            camera_error=runtime.camera_error,
            history_enabled=app.config["LOG_HISTORY"],
            scope="Protótipo de vocabulário limitado. Coleta real e revisão linguística necessárias.",
        )

    @app.get("/api/signs")
    def signs():
        return jsonify(
            signs=runtime.db.get_signs(),
            models=runtime.engine.status(),
            seq_length=SEQ_LENGTH,
        )

    @app.post("/api/signs")
    def create_sign():
        begin_edit()
        try:
            runtime.db.upsert_sign(payload())
            runtime.signs = {s["canonical"]: s for s in runtime.db.get_signs()}
        finally:
            runtime.edit_lock.release()
        return jsonify(ok=True, message="Catálogo atualizado.")

    @app.post("/api/samples/delete")
    def delete_samples():
        begin_edit()
        try:
            label = runtime.db.get_sign(payload().get("label"))["label"]
            # Uma coleta iniciada antes da exclusão não pode repor os dados depois.
            with runtime.clients_lock:
                active = list(runtime.clients.values())
            for state in active:
                with state.lock:
                    if state.capture and state.capture.sign["label"] == label:
                        state.capture = None
                        state.stabilizer.reset()
            for kind in runtime.db.delete_samples(label):
                runtime.engine.invalidate(kind)
        finally:
            runtime.edit_lock.release()
        return jsonify(
            ok=True,
            message="Gravações apagadas. Modelos afetados desativados até novo treinamento.",
        )

    @app.post("/api/train")
    def train():
        data = payload()
        kind = data.get("kind", "both")
        if kind not in ("static", "dynamic", "both"):
            raise ValueError("Tipo de treinamento inválido.")
        begin_edit()
        job_id = secrets.token_hex(8)
        with runtime.job_lock:
            runtime.job = {
                "id": job_id,
                "status": "running",
                "message": "Validando gravações...",
                "results": {},
            }

        def work():
            def progress(message):
                with runtime.job_lock:
                    runtime.job["message"] = message

            try:
                outcomes = {}
                for model_kind in ["static", "dynamic"] if kind == "both" else [kind]:
                    try:
                        runtime.engine.train(model_kind, progress)
                        outcomes[model_kind] = {"trained": True}
                    except ValueError as exc:
                        outcomes[model_kind] = {"trained": False, "error": str(exc)}
                with runtime.job_lock:
                    runtime.job.update(
                        status="done",
                        message="Treinamento encerrado; confira cada resultado.",
                        results=outcomes,
                    )
            except Exception:
                app.logger.exception("Falha no treinamento")
                with runtime.job_lock:
                    runtime.job.update(
                        status="error",
                        message="Treinamento falhou. Consulte o terminal.",
                    )
            finally:
                runtime.edit_lock.release()

        try:
            socketio.start_background_task(work)
        except Exception:
            runtime.edit_lock.release()
            raise
        return jsonify(job_id=job_id), 202

    @app.get("/api/train/status")
    def train_status():
        admin_required()
        with runtime.job_lock:
            return jsonify(runtime.job.copy())

    @app.get("/api/history")
    def history():
        admin_required()
        return jsonify(history=runtime.db.history())

    @app.post("/api/history/clear")
    def clear_history():
        admin_required()
        runtime.db.clear_history()
        return jsonify(ok=True)

    @socketio.on("connect")
    def connect(auth):
        if not isinstance(auth, dict) or not hmac.compare_digest(
            str(auth.get("csrf", "")), session.get("csrf", "!")
        ):
            return False
        with runtime.clients_lock:
            if len(runtime.clients) >= MAX_SESSIONS:
                return False
            runtime.clients[request.sid] = ClientState(session.get("client_id"))

    @socketio.on("disconnect")
    def disconnect(reason=None):
        with runtime.clients_lock:
            state = runtime.clients.pop(request.sid, None)
        if state:
            with state.lock:
                state.close()

    def client():
        with runtime.clients_lock:
            state = runtime.clients.get(request.sid)
        if state is None:
            raise ValueError("Conexão expirada.")
        return state

    @socketio.on("reset_state")
    def reset_state(data=None):
        state = client()
        with state.lock:
            state.reset()
            state.generation = None
        return {"ok": True}

    @socketio.on("configure_threshold")
    def threshold(data):
        if not isinstance(data, dict):
            return {"ok": False, "error": "Configuração inválida."}
        try:
            values = {}
            for key in ("static", "dynamic"):
                val = data.get(key)
                if (
                    type(val) not in (float, int)
                    or not math.isfinite(val)
                    or not 0.5 <= val <= 0.98
                ):
                    raise ValueError("Use limites de 50% a 98%.")
                values[key] = float(val)
            state = client()
            with state.lock:
                state.stabilizer.thresholds = values
            return {"ok": True}
        except ValueError as e:
            return {"ok": False, "error": str(e)}

    @socketio.on("start_capture")
    def start_capture(data):
        try:
            if not runtime.is_admin():
                raise ValueError("Entre em Administração para coletar.")
            if not isinstance(data, dict):
                raise ValueError("Coleta inválida.")
            if runtime.edit_lock.locked():
                raise ValueError("Aguarde o treinamento terminar.")
            if runtime.limited(("capture", session.get("client_id")), 20, 60):
                raise ValueError("Muitas coletas em um minuto.")
            state = client()
            with state.lock:
                if state.capture:
                    raise ValueError("Cancele ou termine a coleta atual.")
                if (
                    state.generation != data.get("generation")
                    or state.last_client_time is None
                ):
                    raise ValueError("Inicie a câmera e aguarde o primeiro frame.")
                state.capture = Capture(
                    runtime.db.get_sign(data.get("label")), data, runtime.clock()
                )
                state.stabilizer.reset()
            return {"ok": True}
        except ValueError as e:
            return {"ok": False, "error": str(e)}

    @socketio.on("cancel_capture")
    def cancel_capture(data=None):
        state = client()
        with state.lock:
            state.capture = None
            state.stabilizer.reset()
        return {"ok": True}

    @socketio.on("process_frame")
    def process_frame(data):
        state = client()
        if not state.lock.acquire(blocking=False):
            return {
                "ok": False,
                "error": "Frame anterior em processamento.",
                "retryable": True,
            }
        try:
            if not isinstance(data, dict):
                raise ValueError("Frame inválido.")
            now = runtime.clock()
            if now - state.last_received < 0.06:
                return {
                    "ok": False,
                    "error": "Reduza a frequência de frames.",
                    "retryable": True,
                }
            state.last_received = now
            generation = data.get("generation")
            timestamp = data.get("timestamp")
            if not isinstance(generation, str) or not 1 <= len(generation) <= 64:
                raise ValueError("Sessão de câmera inválida.")
            if (
                type(timestamp) not in (float, int)
                or not math.isfinite(timestamp)
                or timestamp < 0
            ):
                raise ValueError("Timestamp inválido.")
            if generation != state.generation:
                state.reset()
                state.generation = generation
            if (
                state.last_client_time is not None
                and timestamp <= state.last_client_time
            ):
                raise ValueError("Frame atrasado ou repetido.")
            if (
                state.last_client_time is not None
                and timestamp - state.last_client_time > MAX_FRAME_GAP
            ):
                if state.capture:
                    state.capture = None
                    raise ValueError("Coleta interrompida. Repita a gravação.")
                state.stabilizer.reset()
                if state.extractor:
                    # Um atraso invalida o movimento, sem recriar os grafos de
                    # detecção e provocar outro atraso na máquina mais lenta.
                    state.extractor.reset_motion()
            state.last_client_time = timestamp
            frame = decode_base64_frame(data.get("image"))
            if state.extractor is None:
                try:
                    state.extractor = runtime.extractor_factory()
                    runtime.camera_error = None
                except Exception:
                    runtime.camera_error = "MediaPipe indisponível. Confira Python e requirements.txt no terminal."
                    app.logger.exception("Inicialização do MediaPipe")
                    raise ValueError(runtime.camera_error)
            extracted = state.extractor.process_bgr_frame(frame, timestamp)
            capture_status = {"active": False}
            if state.capture:
                if not runtime.is_admin():
                    state.capture = None
                    raise ValueError(
                        "Sessão administrativa expirada. Coleta cancelada."
                    )
                try:
                    done = state.capture.add(extracted, timestamp, now)
                    capture_status = state.capture.progress()
                    if done:
                        if not runtime.edit_lock.acquire(blocking=False):
                            raise ValueError(
                                "Treinamento iniciado durante a coleta. Repita depois."
                            )
                        try:
                            state.capture.save(runtime.db)
                            runtime.engine.invalidate(state.capture.kind)
                        finally:
                            runtime.edit_lock.release()
                        capture_status = {
                            "active": False,
                            "saved": True,
                            "label": state.capture.sign["label"],
                        }
                        state.capture = None
                    rec = result("collecting", "Coleta em andamento")
                except ValueError:
                    state.capture = None
                    raise
            else:
                rec = state.stabilizer.process_step(
                    extracted, timestamp, runtime.engine, runtime.signs
                )
                if rec["emitted"] and app.config["LOG_HISTORY"]:
                    runtime.db.log_recognition(
                        request.sid, rec["emitted"], rec["score"]
                    )
            state.last_activity = now
            return {
                "ok": True,
                "generation": generation,
                "frame_id": data.get("frame_id"),
                "landmarks": extracted["landmarks"],
                "hands_detected": extracted["hands_detected"],
                "body_detected": extracted["body_detected"],
                "velocity": round(extracted["velocity"], 3),
                "recognition": rec,
                "capture": capture_status,
            }
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception:
            state.capture = None
            app.logger.exception("Falha ao processar frame")
            return {
                "ok": False,
                "error": "Falha no processamento. Confira o terminal e tente novamente.",
            }
        finally:
            state.lock.release()

    return app


def generate_self_signed_cert(cert_path, key_path, hosts):
    """Certificado exclusivo da instalação; confiança deve ser configurada no cliente."""
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    names = []
    for host in set(["localhost", "127.0.0.1", *hosts]):
        try:
            names.append(x509.IPAddress(ipaddress.ip_address(host)))
        except ValueError:
            names.append(x509.DNSName(host))
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "LibrasVision Local")])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=90))
        .add_extension(x509.SubjectAlternativeName(names), False)
        .sign(key, hashes.SHA256())
    )
    Path(key_path).write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    os.chmod(key_path, 0o600)
    Path(cert_path).write_bytes(cert.public_bytes(serialization.Encoding.PEM))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LibrasVision local v2")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--https", action="store_true")
    parser.add_argument("--cert-host", action="append", default=[])
    parser.add_argument("--cert")
    parser.add_argument("--key")
    args = parser.parse_args()
    application = create_app({"SESSION_COOKIE_SECURE": args.https})
    kwargs = {}
    if bool(args.cert) != bool(args.key):
        parser.error("Informe --cert e --key juntos.")
    if args.https:
        if args.cert:
            cert, key = Path(args.cert), Path(args.key)
        else:
            tls = Path(application.config["DATA_DIR"]) / "tls"
            tls.mkdir(parents=True, exist_ok=True)
            cert, key = tls / "cert.pem", tls / "key.pem"
            # Rotação na inicialização inclui o IP atual. Nunca usa os certificados antigos.
            generate_self_signed_cert(cert, key, args.cert_host)
        kwargs["ssl_context"] = (str(cert), str(key))
    if not application.config["ADMIN_TOKEN"]:
        print(
            "Administração desativada. Configure LIBRAS_ADMIN_TOKEN com 32 ou mais caracteres."
        )
    print(
        f"Protótipo local: {'https' if args.https else 'http'}://{args.host}:{args.port}"
    )
    application.extensions["libras_socket"].run(
        application,
        host=args.host,
        port=args.port,
        allow_unsafe_werkzeug=True,
        **kwargs,
    )
