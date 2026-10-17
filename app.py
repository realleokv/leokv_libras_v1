import os
import sys
import base64
import datetime
import ipaddress
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

from config import SEQ_LENGTH, BASE_DIR
from database import (
    init_db,
    get_all_signs,
    add_custom_sign,
    get_model_summaries,
    log_recognition
)
from feature_extractor import HandFeatureExtractor
from ml_engine import LibrasMLEngine
from stabilizer import TemporalStabilizer

app = Flask(__name__)
app.config["SECRET_KEY"] = "libras-ai-secret-key-2026"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

init_db()
extractor = HandFeatureExtractor()
ml_engine = LibrasMLEngine()
stabilizer = TemporalStabilizer()


def decode_base64_frame(data_url: str) -> np.ndarray:
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    img_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


# -------------------- Rotas HTTP / REST API -------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/signs", methods=["GET"])
def api_get_signs():
    return jsonify({
        "signs": get_all_signs(),
        "models": get_model_summaries(),
        "seq_length": SEQ_LENGTH
    })


@app.route("/api/signs", methods=["POST"])
def api_create_sign():
    data = request.get_json(force=True)
    label = data.get("label", "").strip()
    category = data.get("category", "palavra").strip()
    is_dynamic = int(bool(data.get("is_dynamic", False)))
    two_hands = int(bool(data.get("two_hands", False)))
    description = data.get("description", "Sinal cadastrado pelo usuário.").strip()

    if not label:
        return jsonify({"error": "Informe o nome/letra do sinal."}), 400

    add_custom_sign(label, category, is_dynamic, two_hands, description)
    return jsonify({"message": f"Sinal '{label}' salvo com sucesso.", "signs": get_all_signs()})


@app.route("/api/samples/delete", methods=["POST"])
def api_delete_samples():
    data = request.get_json(force=True)
    label = data.get("label", "").strip()
    if not label:
        return jsonify({"error": "Sinal inválido."}), 400
    ml_engine.delete_samples_for_label(label)
    return jsonify({"message": f"Amostras de '{label}' removidas.", "signs": get_all_signs()})


@app.route("/api/train", methods=["POST"])
def api_train_models():
    static_res = ml_engine.train_static_model()
    dynamic_res = ml_engine.train_dynamic_model()
    return jsonify({
        "static": static_res,
        "dynamic": dynamic_res,
        "signs": get_all_signs(),
        "models": get_model_summaries()
    })


@app.route("/api/grammar", methods=["POST"])
def api_libras_grammar():
    """
    Formata a sequência de glosas reconhecidas em Libras (estrutura Tópico-Comentário / SOV)
    para uma leitura fluida em Português sem desrespeitar a glosa original.
    """
    data = request.get_json(force=True)
    tokens = data.get("tokens", [])
    if not tokens:
        return jsonify({"glosa": "", "portugues": ""})

    # Agrupa letras soletradas consecutivas (datilologia) em uma palavra única
    merged = []
    letter_buf = []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            letter_buf.append(t.upper())
        else:
            if letter_buf:
                merged.append("".join(letter_buf))
                letter_buf = []
            merged.append(t)
    if letter_buf:
        merged.append("".join(letter_buf))

    glosa = " ".join([m.upper() for m in merged])

    # Regras gramaticais básicas Libras (Glosa) -> Português estruturado
    pt_words = []
    i = 0
    while i < len(merged):
        w = merged[i]
        w_up = w.upper()
        if w_up == "MEU NOME" and i + 1 < len(merged):
            pt_words.append(f"Meu nome é {merged[i + 1].capitalize()}.")
            i += 2
            continue
        elif w_up == "EU" and i + 1 < len(merged) and merged[i + 1].upper() == "CASA":
            pt_words.append("Eu vou para casa")
            i += 2
            continue
        elif w_up == "EU" and i + 1 < len(merged) and merged[i + 1].upper() == "ÁGUA":
            pt_words.append("Eu quero água")
            i += 2
            continue
        elif w_up == "EU" and i + 1 < len(merged) and merged[i + 1].upper() == "COMIDA":
            pt_words.append("Eu quero comida")
            i += 2
            continue
        elif w_up == "VOCÊ" and i + 1 < len(merged) and merged[i + 1].upper() == "TUDO BEM?":
            pt_words.append("Tudo bem com você?")
            i += 2
            continue
        else:
            pt_words.append(w)
            i += 1

    portugues = " ".join(pt_words)
    if portugues:
        portugues = portugues[0].upper() + portugues[1:]

    return jsonify({"glosa": glosa, "portugues": portugues})


# -------------------- Eventos WebSocket em Tempo Real -------------------- #
@socketio.on("configure_threshold")
def handle_threshold(data):
    val = float(data.get("threshold", 0.72))
    stabilizer.confidence_threshold = max(0.30, min(0.98, val))


@socketio.on("reset_state")
def handle_reset():
    stabilizer.reset()


@socketio.on("process_frame")
def handle_frame(data):
    try:
        frame_b64 = data.get("image")
        if not frame_b64:
            return

        bgr_frame = decode_base64_frame(frame_b64)
        if bgr_frame is None:
            return

        extracted = extractor.process_bgr_frame(bgr_frame)
        rec = stabilizer.process_step(
            features=extracted["features"],
            hands_detected=extracted["hands_detected"],
            velocity=extracted["velocity"],
            ml_engine=ml_engine
        )

        if rec["emitted"]:
            log_recognition(rec["emitted"], rec["confidence"], int(rec["is_dynamic"]))

        emit("frame_result", {
            "landmarks": extracted["landmarks"],
            "hands_detected": extracted["hands_detected"],
            "velocity": round(extracted["velocity"], 4),
            "recognition": rec,
            "features": extracted["features"].tolist() if data.get("return_features") else None
        })
    except Exception as e:
        emit("server_error", {"message": str(e)})


@socketio.on("save_training_sample")
def handle_save_sample(data):
    label = data.get("label", "").strip()
    is_dynamic = bool(data.get("is_dynamic", False))
    payload = data.get("payload", [])

    if not label or not payload:
        emit("sample_saved", {"success": False, "message": "Dados de amostra vazios."})
        return

    if is_dynamic:
        seq = np.array(payload, dtype=np.float32)
        if seq.shape != (SEQ_LENGTH, 146):
            emit("sample_saved", {
                "success": False,
                "message": f"Sequência inválida: esperado ({SEQ_LENGTH}, 146), recebido {seq.shape}."
            })
            return
        total = ml_engine.add_dynamic_sequence(seq, label)
    else:
        total = ml_engine.add_static_samples(payload, label)

    emit("sample_saved", {
        "success": True,
        "label": label,
        "total_for_label": total,
        "is_dynamic": is_dynamic
    })


def generate_self_signed_cert(cert_path: str, key_path: str):
    """Gera certificado SSL autoassinado para permitir acesso à câmera em celulares via HTTPS."""
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, u"LibrasAI-Local")
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(u"localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1"))
            ]),
            critical=False
        )
        .sign(key, hashes.SHA256())
    )
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))


if __name__ == "__main__":
    use_https = "--https" in sys.argv
    port = 5000
    if use_https:
        cert_file = os.path.join(BASE_DIR, "cert.pem")
        key_file = os.path.join(BASE_DIR, "key.pem")
        if not (os.path.exists(cert_file) and os.path.exists(key_file)):
            generate_self_signed_cert(cert_file, key_file)
        print(f"Iniciando servidor HTTPS em https://0.0.0.0:{port}")
        socketio.run(app, host="0.0.0.0", port=port, ssl_context=(cert_file, key_file), allow_unsafe_werkzeug=True)
    else:
        print(f"Iniciando servidor HTTP em http://127.0.0.1:{port}")
        socketio.run(app, host="0.0.0.0", port=port, allow_unsafe_werkzeug=True)