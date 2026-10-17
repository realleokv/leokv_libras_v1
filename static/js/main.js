const socket = io();

// Elementos DOM
const videoEl = document.getElementById("webcam");
const overlayCanvas = document.getElementById("overlay-canvas");
const overlayCtx = overlayCanvas.getContext("2d");
const offscreenCanvas = document.createElement("canvas");
const offscreenCtx = offscreenCanvas.getContext("2d");

const connBadge = document.getElementById("conn-badge");
const fpsBadge = document.getElementById("fps-badge");
const telHands = document.getElementById("tel-hands");
const telVelocity = document.getElementById("tel-velocity");

const predModeTag = document.getElementById("pred-mode-tag");
const predConfText = document.getElementById("pred-conf-text");
const predSignLabel = document.getElementById("pred-sign-label");
const predConfBar = document.getElementById("pred-conf-bar");

const glosaOutput = document.getElementById("glosa-output");
const ptOutput = document.getElementById("pt-output");
const historyList = document.getElementById("history-list");

const selectTrainSign = document.getElementById("select-train-sign");
const selectedSignInfo = document.getElementById("selected-sign-info");
const trainStatusBox = document.getElementById("train-status-box");
const recordingBanner = document.getElementById("recording-banner");
const recProgress = document.getElementById("rec-progress");

// Estado da Aplicação
let stream = null;
let facingMode = "user";
let frameInterval = null;
let awaitingResponse = false;
let frameCount = 0;
let lastFpsTime = performance.now();

let signsCatalog = [];
let seqLength = 20;
let recognizedTokens = [];

// Estado de Coleta de Amostras
let captureState = {
    active: false,
    label: "",
    isDynamic: false,
    targetCount: 0,
    buffer: []
};

// Conexões Anatômicas dos 21 Landmarks (MediaPipe Hands)
const HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],
    [9,13],[13,14],[14,15],[15,16],
    [13,17],[0,17],[17,18],[18,19],[19,20]
];

// -------------------- Navegação de Abas -------------------- //
document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById(btn.dataset.tab).classList.add("active");
    });
});

// -------------------- Conexão WebSocket -------------------- //
socket.on("connect", () => {
    connBadge.textContent = "Conectado";
    connBadge.className = "badge badge-online";
    fetchCatalogAndMetrics();
});

socket.on("disconnect", () => {
    connBadge.textContent = "Desconectado";
    connBadge.className = "badge badge-offline";
});

// -------------------- Controle de Câmera -------------------- //
async function startCamera() {
    if (stream) {
        stopCamera();
    }
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: facingMode,
                width: { ideal: 640 },
                height: { ideal: 480 }
            },
            audio: false
        });
        videoEl.srcObject = stream;
        await videoEl.play();

        overlayCanvas.width = videoEl.videoWidth || 640;
        overlayCanvas.height = videoEl.videoHeight || 480;
        offscreenCanvas.width = 360;
        offscreenCanvas.height = 270;

        document.getElementById("btn-toggle-cam").textContent = "Parar Câmera";
        frameInterval = setInterval(sendFrameToServer, 75); // ~13 FPS para baixa latência
    } catch (err) {
        alert("Erro ao acessar a câmera. Em celulares, acesse via HTTPS. Detalhe: " + err.message);
    }
}

function stopCamera() {
    if (frameInterval) clearInterval(frameInterval);
    frameInterval = null;
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
        stream = null;
    }
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    document.getElementById("btn-toggle-cam").textContent = "Iniciar Câmera";
}

document.getElementById("btn-toggle-cam").addEventListener("click", () => {
    if (stream) stopCamera();
    else startCamera();
});

document.getElementById("btn-switch-cam").addEventListener("click", () => {
    facingMode = facingMode === "user" ? "environment" : "user";
    if (stream) startCamera();
});

function sendFrameToServer() {
    if (!stream || awaitingResponse || videoEl.readyState < 2) return;
    awaitingResponse = true;

    offscreenCtx.drawImage(videoEl, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
    const dataUrl = offscreenCanvas.toDataURL("image/jpeg", 0.65);

    socket.emit("process_frame", {
        image: dataUrl,
        return_features: captureState.active
    });
}

// -------------------- Recebimento dos Resultados em Tempo Real -------------------- //
socket.on("frame_result", (data) => {
    awaitingResponse = false;
    updateFps();

    telHands.textContent = data.hands_detected;
    telVelocity.textContent = data.velocity.toFixed(3);

    drawHandLandmarks(data.landmarks);
    updateRecognitionUI(data.recognition);

    if (captureState.active && data.hands_detected > 0 && data.features) {
        handleFeatureCaptureStep(data.features);
    }
});

function updateFps() {
    frameCount++;
    const now = performance.now();
    if (now - lastFpsTime >= 1000) {
        fpsBadge.textContent = `${frameCount} FPS`;
        frameCount = 0;
        lastFpsTime = now;
    }
}

function drawHandLandmarks(hands) {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    if (!hands || hands.length === 0) return;

    const w = overlayCanvas.width;
    const h = overlayCanvas.height;

    hands.forEach(hand => {
        const pts = hand.points;
        const isLeft = hand.handedness === "Left";
        const strokeColor = isLeft ? "#38bdf8" : "#a855f7";

        overlayCtx.lineWidth = 3;
        overlayCtx.strokeStyle = strokeColor;

        HAND_CONNECTIONS.forEach(([i, j]) => {
            overlayCtx.beginPath();
            overlayCtx.moveTo(pts[i].x * w, pts[i].y * h);
            overlayCtx.lineTo(pts[j].x * w, pts[j].y * h);
            overlayCtx.stroke();
        });

        pts.forEach((p, idx) => {
            overlayCtx.beginPath();
            overlayCtx.arc(p.x * w, p.y * h, idx === 0 ? 6 : 4, 0, 2 * Math.PI);
            overlayCtx.fillStyle = idx % 4 === 0 && idx !== 0 ? "#ffffff" : strokeColor;
            overlayCtx.fill();
        });
    });
}

function updateRecognitionUI(rec) {
    predSignLabel.textContent = rec.display_label;
    predConfText.textContent = `${rec.confidence}%`;
    predConfBar.style.width = `${rec.confidence}%`;

    if (rec.status === "recognized") {
        predModeTag.textContent = rec.is_dynamic ? "Sinal Dinâmico (LSTM)" : "Sinal Estático (RF)";
    } else if (rec.status === "untrained") {
        predModeTag.textContent = "Treinamento Pendente";
    } else {
        predModeTag.textContent = "Monitorando";
    }

    if (rec.emitted) {
        recognizedTokens.push(rec.emitted);
        refreshGrammarTranslation();
        prependHistory(rec.emitted, rec.confidence, rec.is_dynamic);
    }
}

function prependHistory(label, conf, isDynamic) {
    const li = document.createElement("li");
    const time = new Date().toLocaleTimeString("pt-BR");
    li.innerHTML = `
        <span><strong>${label}</strong> <small>(${isDynamic ? "Dinâmico" : "Estático"})</small></span>
        <span>${conf}% • ${time}</span>
    `;
    historyList.prepend(li);
    while (historyList.children.length > 15) {
        historyList.removeChild(historyList.lastChild);
    }
}

// -------------------- Formador de Texto e Gramática Libras -------------------- //
async function refreshGrammarTranslation() {
    if (recognizedTokens.length === 0) {
        glosaOutput.textContent = "Nenhum sinal emitido ainda...";
        ptOutput.value = "";
        return;
    }
    const res = await fetch("/api/grammar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tokens: recognizedTokens })
    });
    const data = await res.json();
    glosaOutput.textContent = data.glosa;
    ptOutput.value = data.portugues;
}

document.getElementById("btn-space").addEventListener("click", () => {
    recognizedTokens.push(" ");
    refreshGrammarTranslation();
});

document.getElementById("btn-backspace").addEventListener("click", () => {
    recognizedTokens.pop();
    socket.emit("reset_state");
    refreshGrammarTranslation();
});

document.getElementById("btn-clear").addEventListener("click", () => {
    recognizedTokens = [];
    socket.emit("reset_state");
    refreshGrammarTranslation();
});

document.getElementById("btn-copy").addEventListener("click", () => {
    navigator.clipboard.writeText(ptOutput.value || "");
});

document.getElementById("btn-speak").addEventListener("click", () => {
    const text = ptOutput.value.trim();
    if (!text || !("speechSynthesis" in window)) return;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "pt-BR";
    window.speechSynthesis.speak(utterance);
});

document.getElementById("slider-threshold").addEventListener("input", (e) => {
    const val = parseInt(e.target.value, 10);
    document.getElementById("lbl-threshold").textContent = `${val}%`;
    socket.emit("configure_threshold", { threshold: val / 100.0 });
});

// -------------------- Catálogo, Coleta de Amostras e Treinamento -------------------- //
async function fetchCatalogAndMetrics() {
    const res = await fetch("/api/signs");
    const data = await res.json();
    signsCatalog = data.signs || [];
    seqLength = data.seq_length || 20;

    populateSelectAndTable(signsCatalog, data.models || {});
}

function populateSelectAndTable(signs, models) {
    const prevValue = selectTrainSign.value;
    selectTrainSign.innerHTML = "";
    signs.forEach(s => {
        const opt = document.createElement("option");
        opt.value = s.label;
        opt.textContent = `${s.label} (${s.is_dynamic ? "Dinâmico - Movimento" : "Estático"}) — [${s.samples_count} amostras]`;
        selectTrainSign.appendChild(opt);
    });
    if (prevValue) selectTrainSign.value = prevValue;
    updateSelectedSignInfo();

    // Resumo global de modelos
    if (models.static_rf) {
        document.getElementById("metric-static-acc").textContent = `${(models.static_rf.global_accuracy * 100).toFixed(1)}%`;
        document.getElementById("metric-static-meta").textContent = `${models.static_rf.trained_classes} classes • ${models.static_rf.total_samples} amostras`;
    }
    if (models.dynamic_lstm) {
        document.getElementById("metric-dynamic-acc").textContent = `${(models.dynamic_lstm.global_accuracy * 100).toFixed(1)}%`;
        document.getElementById("metric-dynamic-meta").textContent = `${models.dynamic_lstm.trained_classes} classes • ${models.dynamic_lstm.total_samples} sequências`;
    }

    renderSignsTable(signs);
}

function updateSelectedSignInfo() {
    const sign = signsCatalog.find(s => s.label === selectTrainSign.value);
    if (!sign) return;
    selectedSignInfo.innerHTML = `
        <strong>${sign.label}</strong> (${sign.is_dynamic ? "Dinâmico / LSTM 20 frames" : "Estático / Scikit-learn"})<br>
        <span>${sign.description}</span>
    `;
}

selectTrainSign.addEventListener("change", updateSelectedSignInfo);

document.getElementById("btn-capture-samples").addEventListener("click", () => {
    if (!stream) {
        alert("Inicie a câmera na aba 'Reconhecimento ao Vivo' antes de capturar amostras!");
        return;
    }
    const sign = signsCatalog.find(s => s.label === selectTrainSign.value);
    if (!sign) return;

    const isDyn = Boolean(sign.is_dynamic);
    const target = isDyn ? seqLength : parseInt(document.getElementById("input-burst-count").value, 10) || 35;

    captureState = {
        active: true,
        label: sign.label,
        isDynamic: isDyn,
        targetCount: target,
        buffer: []
    };

    recordingBanner.classList.remove("hidden");
    recProgress.textContent = `0/${target}`;
    trainStatusBox.textContent = `Capturando '${sign.label}'... Mantenha a mão visível na câmera.`;
});

function handleFeatureCaptureStep(featuresArray) {
    captureState.buffer.push(featuresArray);
    recProgress.textContent = `${captureState.buffer.length}/${captureState.targetCount}`;

    if (captureState.buffer.length >= captureState.targetCount) {
        const payload = captureState.buffer.slice(0, captureState.targetCount);
        const label = captureState.label;
        const isDynamic = captureState.isDynamic;

        captureState.active = false;
        recordingBanner.classList.add("hidden");

        socket.emit("save_training_sample", {
            label: label,
            is_dynamic: isDynamic,
            payload: payload
        });
    }
}

socket.on("sample_saved", (res) => {
    if (res.success) {
        trainStatusBox.textContent = `Amostra salva para '${res.label}'! Total acumulado para este sinal: ${res.total_for_label}.`;
        fetchCatalogAndMetrics();
    } else {
        trainStatusBox.textContent = `Erro ao salvar amostra: ${res.message}`;
    }
});

document.getElementById("btn-train-models").addEventListener("click", async () => {
    trainStatusBox.textContent = "Treinando modelos Scikit-learn e PyTorch LSTM... Aguarde.";
    const res = await fetch("/api/train", { method: "POST" });
    const data = await res.json();

    trainStatusBox.innerHTML = `
        <strong>Estático:</strong> ${data.static.message}<br>
        <strong>Dinâmico:</strong> ${data.dynamic.message}
    `;
    populateSelectAndTable(data.signs || [], data.models || {});
});

document.getElementById("form-new-sign").addEventListener("submit", async (e) => {
    e.preventDefault();
    const payload = {
        label: document.getElementById("new-sign-label").value,
        category: document.getElementById("new-sign-category").value,
        is_dynamic: parseInt(document.getElementById("new-sign-dynamic").value, 10),
        two_hands: 0,
        description: document.getElementById("new-sign-desc").value
    };
    const res = await fetch("/api/signs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (res.ok) {
        e.target.reset();
        fetchCatalogAndMetrics();
        trainStatusBox.textContent = data.message;
    }
});

function renderSignsTable(signs) {
    const filter = (document.getElementById("filter-signs-input").value || "").toLowerCase();
    const tbody = document.getElementById("signs-table-body");
    tbody.innerHTML = "";

    signs
        .filter(s => s.label.toLowerCase().includes(filter) || s.description.toLowerCase().includes(filter))
        .forEach(s => {
            const tr = document.createElement("tr");
            const prec = s.precision_score !== null ? `${(s.precision_score * 100).toFixed(1)}%` : "Sem dados";
            const f1 = s.f1_score !== null ? `${(s.f1_score * 100).toFixed(1)}%` : "—";

            tr.innerHTML = `
                <td><strong>${s.label}</strong></td>
                <td>${s.category}</td>
                <td>${s.is_dynamic ? "LSTM (Movimento)" : "Random Forest (Estático)"}</td>
                <td>${s.description}</td>
                <td>${s.samples_count}</td>
                <td>${prec}</td>
                <td>${f1}</td>
                <td>
                    <button class="btn btn-outline btn-sm" onclick="deleteSignSamples('${s.label}')">Limpar Amostras</button>
                </td>
            `;
            tbody.appendChild(tr);
        });
}

document.getElementById("filter-signs-input").addEventListener("input", () => {
    renderSignsTable(signsCatalog);
});

async function deleteSignSamples(label) {
    if (!confirm(`Deseja apagar todas as amostras gravadas do sinal '${label}'?`)) return;
    await fetch("/api/samples/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label })
    });
    fetchCatalogAndMetrics();
}