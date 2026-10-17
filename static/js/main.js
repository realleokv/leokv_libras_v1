/* Fluxo serial com confirmação de cada frame e cancelamento por sessão de câmera. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const video = $("webcam"),
    overlay = $("overlay-canvas"),
    ctx = overlay.getContext("2d");
  const offscreen = document.createElement("canvas"),
    offctx = offscreen.getContext("2d");
  let csrf = "",
    admin = false,
    socket = null,
    signs = [],
    modelStatus = {};
  let stream = null,
    starting = false,
    facing = "user",
    cameraSerial = 0,
    generation = "",
    timer = null;
  let frameId = 0,
    frames = 0,
    fpsStart = performance.now(),
    readyFrame = false,
    failures = 0;
  let tokens = [],
    reviewDirty = false,
    currentLabel = null,
    capturePending = false,
    captureActive = false;
  let captureSerial = 0,
    training = false,
    pollTimer = null;
  const connections = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [5, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [9, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [13, 17],
    [0, 17],
    [17, 18],
    [18, 19],
    [19, 20],
  ];
  const notice = (message) => {
    $("notice").textContent = message;
  };
  const safe =
    (fn) =>
    async (...args) => {
      try {
        await fn(...args);
      } catch (e) {
        notice(e.message || "Não foi possível concluir a ação.");
      }
    };
  const node = (tag, text, cls) => {
    const el = document.createElement(tag);
    if (text !== undefined) el.textContent = text;
    if (cls) el.className = cls;
    return el;
  };

  async function api(path, data) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 12000);
    try {
      const response = await fetch(path, {
        method: data === undefined ? "GET" : "POST",
        signal: controller.signal,
        headers:
          data === undefined
            ? {}
            : { "Content-Type": "application/json", "X-CSRF-Token": csrf },
        body: data === undefined ? undefined : JSON.stringify(data),
      });
      const body = await response.json();
      if (!response.ok)
        throw new Error(body.error || `Erro ${response.status}`);
      return body;
    } catch (e) {
      if (e.name === "AbortError")
        throw new Error("O servidor demorou a responder. Tente novamente.");
      throw e;
    } finally {
      clearTimeout(timeout);
    }
  }

  function requestSocket(event, data = {}, timeout = 6000) {
    return new Promise((resolve, reject) => {
      if (!socket?.connected)
        return reject(new Error("Servidor desconectado."));
      socket.timeout(timeout).emit(event, data, (error, response) => {
        if (error)
          reject(new Error("Sem resposta do servidor. O envio será retomado."));
        else if (!response?.ok)
          reject(
            Object.assign(new Error(response?.error || "Resposta inválida."), {
              retryable: response?.retryable,
            }),
          );
        else resolve(response);
      });
    });
  }

  function setTab(button) {
    document.querySelectorAll('[role="tab"]').forEach((tab) => {
      const selected = tab === button;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
      tab.classList.toggle("active", selected);
      const panel = $(tab.dataset.tab);
      panel.hidden = !selected;
      panel.classList.toggle("active", selected);
    });
  }
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => setTab(tab));
    tab.addEventListener("keydown", (e) => {
      let target;
      if (e.key === "ArrowRight") target = tabs[(i + 1) % tabs.length];
      if (e.key === "ArrowLeft")
        target = tabs[(i + tabs.length - 1) % tabs.length];
      if (e.key === "Home") target = tabs[0];
      if (e.key === "End") target = tabs.at(-1);
      if (target) {
        e.preventDefault();
        setTab(target);
        target.focus();
      }
    });
  });

  function syncAdmin() {
    $("form-login").hidden = admin;
    $("btn-logout").hidden = !admin;
    $("auth-status").textContent = admin
      ? "Administração ativa por até uma hora."
      : "Entre para coletar, alterar dados ou treinar.";
    ["capture-fields", "train-fields", "sign-fields"].forEach((id) => {
      $(id).disabled = !admin || training;
    });
    $("btn-clear-server-history").disabled = !admin;
    renderTable();
  }
  $("form-login").addEventListener(
    "submit",
    safe(async (e) => {
      e.preventDefault();
      const token = $("admin-token").value;
      $("admin-token").value = "";
      await api("/api/login", { token });
      admin = true;
      syncAdmin();
      notice("Administração liberada.");
      await checkTraining();
    }),
  );
  $("btn-logout").addEventListener(
    "click",
    safe(async () => {
      await cancelCapture();
      await api("/api/logout", {});
      admin = false;
      clearTimeout(pollTimer);
      syncAdmin();
    }),
  );

  async function initialize() {
    if (typeof io !== "function")
      throw new Error(
        "Não foi possível carregar a conexão. Confira static/vendor no servidor.",
      );
    const state = await api("/api/session");
    csrf = state.csrf;
    admin = state.admin;
    syncAdmin();
    if (!state.admin_configured)
      $("auth-status").textContent =
        "Administração desativada. Defina LIBRAS_ADMIN_TOKEN no servidor conforme o README.";
    socket = io({
      auth: { csrf },
      autoConnect: false,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
    });
    socket.on(
      "connect",
      safe(async () => {
        $("conn-badge").textContent = "Conectado";
        $("conn-badge").className = "badge badge-online";
        $("btn-toggle-cam").disabled = starting;
        $("btn-switch-cam").disabled = starting;
        readyFrame = false;
        failures = 0;
        generation = crypto.randomUUID();
        await updateThresholds();
        await fetchCatalog();
        if (stream) scheduleFrame(0);
        if (admin) await checkTraining();
      }),
    );
    socket.on("disconnect", () => {
      clearTimeout(timer);
      readyFrame = false;
      captureSerial++;
      capturePending = captureActive = false;
      $("recording-banner").classList.add("hidden");
      $("capture-status").textContent =
        "Conexão interrompida. Repita a coleta após reconectar.";
      $("conn-badge").textContent = "Desconectado";
      $("conn-badge").className = "badge badge-offline";
      $("btn-toggle-cam").disabled = !stream;
      $("fps-badge").textContent = "0 FPS";
    });
    socket.on("connect_error", () =>
      notice(
        "Falha de conexão ou limite de usuários atingido. Recarregue se a sessão expirou.",
      ),
    );
    socket.connect();
  }

  async function startCamera() {
    if (starting) return;
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia)
      throw new Error(
        "Use localhost no computador ou HTTPS confiável no celular para acessar a câmera.",
      );
    stopCamera();
    starting = true;
    const serial = cameraSerial;
    $("btn-toggle-cam").disabled = $("btn-switch-cam").disabled = true;
    let acquired = null;
    try {
      acquired = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: facing },
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      });
      if (serial !== cameraSerial) {
        acquired.getTracks().forEach((t) => t.stop());
        return;
      }
      stream = acquired;
      video.srcObject = stream;
      await video.play();
      const w = video.videoWidth,
        h = video.videoHeight;
      if (!w || !h) throw new Error("A câmera não retornou uma imagem válida.");
      overlay.width = w;
      overlay.height = h;
      const ratio = Math.min(1, 640 / Math.max(w, h));
      offscreen.width = Math.round(w * ratio);
      offscreen.height = Math.round(h * ratio);
      $("viewport").style.aspectRatio = `${w}/${h}`;
      $("viewport").classList.toggle("mirrored", facing === "user");
      generation = crypto.randomUUID();
      readyFrame = false;
      failures = 0;
      stream.getVideoTracks()[0].addEventListener("ended", () => {
        if (serial === cameraSerial) {
          stopCamera();
          notice("A câmera foi desconectada.");
        }
      });
      $("btn-toggle-cam").textContent = "Parar câmera";
      scheduleFrame(0);
    } catch (e) {
      if (acquired) acquired.getTracks().forEach((t) => t.stop());
      stopCamera();
      const messages = {
        NotAllowedError:
          "A câmera foi bloqueada. Confira a permissão do navegador e o HTTPS.",
        NotFoundError: "Nenhuma câmera foi encontrada.",
        NotReadableError: "A câmera está ocupada por outro aplicativo.",
      };
      throw new Error(messages[e.name] || e.message);
    } finally {
      starting = false;
      $("btn-toggle-cam").disabled = !socket?.connected && !stream;
      $("btn-switch-cam").disabled = false;
    }
  }
  function stopCamera() {
    cameraSerial++;
    captureSerial++;
    clearTimeout(timer);
    timer = null;
    if (stream) stream.getTracks().forEach((track) => track.stop());
    stream = null;
    video.srcObject = null;
    readyFrame = false;
    capturePending = captureActive = false;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    $("recording-banner").classList.add("hidden");
    $("btn-toggle-cam").textContent = "Iniciar câmera";
    $("fps-badge").textContent = "0 FPS";
    $("tel-hands").textContent = "0";
    $("tel-body").textContent = "não detectado";
    $("capture-status").textContent = "Nenhuma gravação em andamento.";
    currentLabel = null;
    $("btn-accept").disabled = true;
    if (socket?.connected)
      requestSocket("reset_state").catch((e) => notice(e.message));
  }
  $("btn-toggle-cam").addEventListener(
    "click",
    safe(async () => {
      if (stream) stopCamera();
      else await startCamera();
    }),
  );
  $("btn-switch-cam").addEventListener(
    "click",
    safe(async () => {
      if (starting) return;
      facing = facing === "user" ? "environment" : "user";
      if (stream) await startCamera();
    }),
  );
  window.addEventListener("pagehide", stopCamera);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && stream) {
      stopCamera();
      notice(
        "Câmera pausada ao sair da página. Inicie novamente para continuar.",
      );
    }
  });
  function scheduleFrame(delay = 75) {
    clearTimeout(timer);
    if (stream && socket?.connected) timer = setTimeout(sendFrame, delay);
  }
  async function sendFrame() {
    const serial = cameraSerial,
      gen = generation;
    try {
      if (!stream || !socket?.connected || video.readyState < 2) return;
      offctx.drawImage(video, 0, 0, offscreen.width, offscreen.height);
      const response = await requestSocket("process_frame", {
        image: offscreen.toDataURL("image/jpeg", 0.8),
        timestamp: performance.now() / 1000,
        generation: gen,
        frame_id: ++frameId,
      });
      if (
        serial !== cameraSerial ||
        gen !== generation ||
        response.generation !== gen
      )
        return;
      readyFrame = true;
      failures = 0;
      frames++;
      if (performance.now() - fpsStart >= 1000) {
        $("fps-badge").textContent =
          `${Math.round((frames * 1000) / (performance.now() - fpsStart))} FPS`;
        frames = 0;
        fpsStart = performance.now();
      }
      $("tel-hands").textContent = response.hands_detected;
      $("tel-body").textContent = response.body_detected
        ? "detectado"
        : "não detectado";
      $("tel-velocity").textContent = response.velocity.toFixed(2);
      drawHands(response.landmarks);
      showRecognition(response.recognition);
      captureActive = Boolean(response.capture.active);
      if (response.capture.active) {
        $("recording-banner").textContent =
          `Gravando ${response.capture.label}: ${response.capture.current}/${response.capture.target}`;
        $("recording-banner").classList.remove("hidden");
      } else if (!capturePending) $("recording-banner").classList.add("hidden");
      if (response.capture.saved) {
        $("capture-status").textContent =
          `Gravação de ${response.capture.label} salva. Faça uma nova execução independente.`;
        await fetchCatalog();
      }
    } catch (e) {
      if (serial !== cameraSerial || gen !== generation) return;
      if (!e.retryable) {
        failures++;
        capturePending = captureActive = false;
        captureSerial++;
        $("recording-banner").classList.add("hidden");
        notice(e.message);
      }
      if (failures >= 3) {
        stopCamera();
        notice(
          `${e.message} A câmera foi pausada após três falhas. Confira o servidor e inicie novamente.`,
        );
      }
    } finally {
      if (serial === cameraSerial && gen === generation)
        scheduleFrame(failures ? 1000 : 75);
    }
  }
  function drawHands(hands) {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    for (const hand of hands || []) {
      ctx.strokeStyle = hand.handedness === "Left" ? "#38bdf8" : "#e9a8ff";
      ctx.fillStyle = ctx.strokeStyle;
      ctx.lineWidth = 3;
      for (const [a, b] of connections) {
        ctx.beginPath();
        ctx.moveTo(
          hand.points[a].x * overlay.width,
          hand.points[a].y * overlay.height,
        );
        ctx.lineTo(
          hand.points[b].x * overlay.width,
          hand.points[b].y * overlay.height,
        );
        ctx.stroke();
      }
      for (const p of hand.points) {
        ctx.beginPath();
        ctx.arc(p.x * overlay.width, p.y * overlay.height, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
  function showRecognition(rec) {
    $("pred-sign-label").textContent = rec.display_label;
    $("pred-conf-text").textContent = `${rec.score}%`;
    $("pred-conf-bar").value = rec.score;
    $("pred-mode-tag").textContent =
      rec.status === "recognized"
        ? rec.is_dynamic
          ? "Com movimento"
          : "Estático"
        : "Aguardando um resultado estável";
    currentLabel = rec.status === "recognized" ? rec.display_label : null;
    $("btn-accept").disabled = !currentLabel;
    if (rec.emitted) {
      $("recognition-announcement").textContent = `Sinal: ${rec.emitted}`;
      if ($("auto-accept").checked) addToken(rec.emitted);
    }
  }
  function addToken(label) {
    if (tokens.length >= 300) {
      notice(
        "Limite de 300 sinais. Copie a revisão e limpe o texto para continuar.",
      );
      return;
    }
    const sign = signs.find((s) => s.label === label);
    tokens.push({
      kind:
        sign?.category === "alfabeto" && [...label].length === 1
          ? "letter"
          : "word",
      text: label,
    });
    updateText();
    $("history-list").prepend(
      node("li", `${label} · ${new Date().toLocaleTimeString("pt-BR")}`),
    );
    while ($("history-list").children.length > 15)
      $("history-list").lastChild.remove();
  }
  function updateText() {
    const text = LibrasText.formatTokens(tokens);
    $("auto-output").value = text;
    if (!reviewDirty) $("pt-output").value = text;
    $("edit-status").textContent = reviewDirty
      ? "Sua revisão está preservada. Novos sinais aparecem somente no texto reconhecido."
      : "Acompanha o reconhecimento até você começar a editar.";
  }
  $("pt-output").addEventListener("input", () => {
    reviewDirty = true;
    updateText();
  });
  $("btn-sync-text").addEventListener("click", () => {
    if (
      reviewDirty &&
      !confirm("Substituir sua revisão pelo texto reconhecido?")
    )
      return;
    reviewDirty = false;
    updateText();
  });
  $("btn-accept").addEventListener("click", () => {
    if (currentLabel) addToken(currentLabel);
  });
  $("btn-space").addEventListener("click", () => {
    if (
      tokens.length < 300 &&
      tokens.length &&
      tokens.at(-1).kind !== "separator"
    )
      tokens.push({ kind: "separator" });
    updateText();
  });
  $("btn-backspace").addEventListener("click", () => {
    tokens.pop();
    updateText();
  });
  $("btn-clear").addEventListener("click", () => {
    tokens = [];
    updateText();
    notice(
      reviewDirty
        ? "Reconhecimento limpo. Sua revisão foi preservada."
        : "Texto limpo.",
    );
  });
  $("btn-copy").addEventListener(
    "click",
    safe(async () => {
      if (!navigator.clipboard?.writeText)
        throw new Error(
          "Selecione o texto de revisão e copie manualmente neste navegador.",
        );
      await navigator.clipboard.writeText($("pt-output").value);
      notice("Revisão copiada.");
    }),
  );
  $("btn-speak").addEventListener("click", () => {
    if (!("speechSynthesis" in window))
      return notice("Este navegador não oferece leitura em voz.");
    speechSynthesis.cancel();
    const speech = new SpeechSynthesisUtterance($("pt-output").value);
    speech.lang = "pt-BR";
    speech.onerror = () => notice("Não foi possível reproduzir a voz.");
    speechSynthesis.speak(speech);
  });
  async function updateThresholds() {
    await requestSocket("configure_threshold", {
      static: Number($("threshold-static").value) / 100,
      dynamic: Number($("threshold-dynamic").value) / 100,
    });
  }
  ["static", "dynamic"].forEach((kind) =>
    $("threshold-" + kind).addEventListener(
      "change",
      safe(async () => {
        $("threshold-" + kind + "-value").textContent =
          $("threshold-" + kind).value + "%";
        await updateThresholds();
      }),
    ),
  );

  async function fetchCatalog() {
    const data = await api("/api/signs");
    signs = data.signs;
    modelStatus = data.models;
    const select = $("select-train-sign"),
      previous = select.value;
    select.replaceChildren();
    for (const s of signs) {
      const option = node(
        "option",
        s.label === "__NONE__" ? "Nenhum sinal / transições" : s.label,
      );
      option.value = s.label;
      select.append(option);
    }
    if (signs.some((s) => s.label === previous)) select.value = previous;
    updateSelectedSign();
    renderTable();
    for (const kind of ["static", "dynamic"]) {
      const state = modelStatus[kind];
      $("metric-" + kind).textContent = state?.ready
        ? `Acurácia em gravações isoladas: ${(state.metrics.accuracy * 100).toFixed(1)}%. ${state.metrics.test_recordings} gravações de teste; ${state.metrics.people} pessoas no conjunto. ${state.classes.length} classes incluindo controle.`
        : state?.error ||
          "Sem modelo v2 disponível. Faça coleta real e treinamento.";
    }
  }
  function updateSelectedSign() {
    const sign = signs.find((s) => s.label === $("select-train-sign").value);
    if (!sign) return;
    $("selected-sign-info").textContent =
      `${sign.description} ${sign.two_hands ? "Exige duas mãos. " : ""}${sign.requires_body ? "Enquadre rosto e ombros. " : ""}${sign.reviewed ? "Revisão registrada." : "Revisão linguística pendente."}`;
    $("capture-kind").disabled = sign.label !== "__NONE__";
    if (sign.label !== "__NONE__")
      $("capture-kind").value = sign.is_dynamic ? "dynamic" : "static";
  }
  $("select-train-sign").addEventListener("change", updateSelectedSign);
  async function cancelCapture() {
    captureSerial++;
    capturePending = captureActive = false;
    $("recording-banner").classList.add("hidden");
    $("capture-status").textContent = "Gravação cancelada.";
    if (socket?.connected) await requestSocket("cancel_capture");
  }
  $("btn-cancel-capture").addEventListener("click", safe(cancelCapture));
  $("btn-capture-samples").addEventListener(
    "click",
    safe(async () => {
      if (!stream || !readyFrame)
        throw new Error("Inicie a câmera e aguarde o primeiro resultado.");
      if (capturePending || captureActive)
        throw new Error("Termine ou cancele a gravação atual.");
      const data = {
        label: $("select-train-sign").value,
        kind: $("capture-kind").value,
        signer: $("signer-code").value.trim(),
        session: $("session-code").value.trim(),
        count: Number($("input-burst-count").value),
        generation,
      };
      if (!data.signer || !data.session)
        throw new Error("Preencha os códigos da pessoa e da sessão.");
      if (!Number.isInteger(data.count) || data.count < 6 || data.count > 30)
        throw new Error("Use de 6 a 30 frames estáticos.");
      capturePending = true;
      const serial = ++captureSerial;
      try {
        for (let count = 3; count > 0; count--) {
          $("capture-status").textContent =
            `Prepare o sinal ${data.label}. Começa em ${count}...`;
          await new Promise((resolve) => setTimeout(resolve, 1000));
          if (serial !== captureSerial) return;
        }
        await requestSocket("start_capture", data);
        if (serial !== captureSerial) return;
        captureActive = true;
        $("capture-status").textContent =
          "Gravando. Mantenha o enquadramento e execute o sinal.";
      } finally {
        if (serial === captureSerial) capturePending = false;
      }
    }),
  );

  async function checkTraining() {
    clearTimeout(pollTimer);
    const job = await api("/api/train/status");
    training = job.status === "running";
    syncAdmin();
    const details = Object.entries(job.results || {})
      .map(([kind, res]) => `${kind}: ${res.trained ? "concluído" : res.error}`)
      .join(" | ");
    $("train-status-box").textContent =
      job.message + (details ? " " + details : "");
    if (training)
      pollTimer = setTimeout(
        () =>
          checkTraining().catch((e) => {
            notice(e.message);
            pollTimer = setTimeout(() => {
              if (admin) checkTraining().catch((err) => notice(err.message));
            }, 3000);
          }),
        1200,
      );
    else await fetchCatalog();
  }
  $("btn-train-models").addEventListener(
    "click",
    safe(async () => {
      if (training) return;
      await cancelCapture();
      training = true;
      syncAdmin();
      try {
        await api("/api/train", { kind: $("train-kind").value });
        await checkTraining();
      } catch (e) {
        training = false;
        syncAdmin();
        throw e;
      }
    }),
  );
  $("form-new-sign").addEventListener(
    "submit",
    safe(async (e) => {
      e.preventDefault();
      await api("/api/signs", {
        label: $("new-sign-label").value,
        category: $("new-sign-category").value,
        description: $("new-sign-desc").value,
        is_dynamic: $("new-sign-dynamic").checked,
        two_hands: $("new-sign-two").checked,
        requires_body: $("new-sign-body").checked,
        reference_url: $("new-sign-reference").value,
        reviewer: $("new-sign-reviewer").value,
        reviewed: $("new-sign-reviewed").checked,
      });
      e.target.reset();
      await fetchCatalog();
      notice("Sinal salvo no catálogo.");
    }),
  );
  function renderTable() {
    const filter = $("filter-signs-input").value.toLocaleLowerCase("pt-BR"),
      body = $("signs-table-body");
    body.replaceChildren();
    for (const s of signs.filter((sign) =>
      `${sign.label} ${sign.description}`
        .toLocaleLowerCase("pt-BR")
        .includes(filter),
    )) {
      const row = node("tr"),
        info = node("td");
      info.append(
        node("strong", s.label === "__NONE__" ? "Nenhum sinal" : s.label),
        node("p", s.description),
        node(
          "small",
          s.reviewed
            ? `Revisão: ${s.reviewer}`
            : "Revisão linguística pendente",
        ),
      );
      if (s.reference_url) {
        const link = node("a", "Abrir referência");
        link.href = s.reference_url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        info.append(node("br"), link);
      }
      row.append(
        info,
        node("td", s.recordings_count),
        node("td", s.signers_count),
      );
      const kind = s.is_dynamic ? "dynamic" : "static",
        model = modelStatus[kind];
      const available = model?.ready && model.classes.includes(s.label),
        metric = available ? model.metrics.report[s.label] : null;
      row.append(
        node(
          "td",
          available
            ? s.is_dynamic
              ? "Movimento disponível"
              : "Estático disponível"
            : "Indisponível",
        ),
      );
      row.append(
        node(
          "td",
          metric
            ? `Precisão ${(metric.precision * 100).toFixed(1)}%; F1 ${(metric["f1-score"] * 100).toFixed(1)}%; teste n=${metric.support}`
            : "Sem avaliação atual",
        ),
      );
      const actions = node("td"),
        button = node("button", "Apagar gravações", "btn btn-outline btn-sm");
      button.disabled = !admin || training || !s.recordings_count;
      button.addEventListener(
        "click",
        safe(async () => {
          if (
            !confirm(
              `Apagar todas as gravações de “${s.label}”? O modelo afetado será desativado até novo treino.`,
            )
          )
            return;
          await api("/api/samples/delete", { label: s.label });
          await fetchCatalog();
        }),
      );
      actions.append(button);
      row.append(actions);
      body.append(row);
    }
  }
  $("filter-signs-input").addEventListener("input", renderTable);
  $("btn-clear-server-history").addEventListener(
    "click",
    safe(async () => {
      if (!confirm("Apagar o histórico salvo no servidor?")) return;
      await api("/api/history/clear", {});
      notice("Histórico do servidor apagado.");
    }),
  );
  initialize().catch((e) => notice(e.message));
})();
