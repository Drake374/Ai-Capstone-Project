//Config
const API_URL = "/recognize";      // same-origin; change if backend is on a different host
const CAPTURE_INTERVAL_MS = 300;   // how often to send a frame (ms)
const MAX_FPS_BAR = 5;             // 5 fps = full bar

//State
let stream = null;
let intervalId = null;
let running = false;
let totalDetected = 0;
let frameCount = 0;
let fpsTimer = Date.now();
let currentFps = 0;
let threshold = 0.70;

//DOM refs
const video     = document.getElementById("video");
const overlay   = document.getElementById("overlay");
const ctx       = overlay.getContext("2d");
const btnStart  = document.getElementById("btn-start");
const btnStop   = document.getElementById("btn-stop");
const scanLine  = document.getElementById("scan-line");
const placeholder = document.getElementById("placeholder");
const dot       = document.getElementById("status-dot");
const statusTxt = document.getElementById("status-text");
const logEl     = document.getElementById("log");
const thresholdInput = document.getElementById("threshold");
const thresholdVal   = document.getElementById("threshold-val");

//Logging
function log(msg, type = "info") {
  const ts = new Date().toLocaleTimeString("en", { hour12: false });
  const entry = document.createElement("div");
  entry.className = "entry";
  entry.innerHTML = `<span class="ts">${ts}</span><span class="msg ${type === "ok" ? "ok" : type === "err" ? "err" : ""}">${msg}</span>`;
  logEl.prepend(entry);
  // Keep log max 40 entries
  while (logEl.children.length > 40) logEl.removeChild(logEl.lastChild);
}

//Threshold
thresholdInput.addEventListener("input", () => {
  threshold = parseFloat(thresholdInput.value);
  thresholdVal.textContent = threshold.toFixed(2);
});

//Status indicator
function setStatus(state) {
  dot.className = "dot " + state;
  statusTxt.textContent = state === "live" ? "LIVE" : state === "error" ? "ERROR" : "OFFLINE";
}

//Resize overlay to match video
function syncCanvas() {
  overlay.width  = video.videoWidth;
  overlay.height = video.videoHeight;
}

//Draw bounding boxes
function drawFaces(faces) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  faces.forEach(face => {
    const [x1, y1, x2, y2] = face.box;
    const w = x2 - x1;
    const h = y2 - y1;
    const isUnknown = face.name === "Unknown";
    const color = "#4ddd00";

    // Box
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, w, h);

    // Corner accents
    const cs = 14; // corner size
    ctx.lineWidth = 3;
    [[x1, y1, 1, 1], [x2, y1, -1, 1], [x1, y2, 1, -1], [x2, y2, -1, -1]].forEach(([cx, cy, dx, dy]) => {
      ctx.beginPath();
      ctx.moveTo(cx + dx * cs, cy);
      ctx.lineTo(cx, cy);
      ctx.lineTo(cx, cy + dy * cs);
      ctx.stroke();
    });

    // Label background
    const label = `${face.name}  ${(face.confidence * 100).toFixed(0)}%`;
    ctx.font = "bold 13px 'Share Tech Mono', monospace";
    const textW = ctx.measureText(label).width;
    const labelX = x1;
    const labelY = y1 - 8;

    ctx.fillStyle = color;
    ctx.fillRect(labelX - 2, labelY - 16, textW + 10, 20);

    // Label text
    ctx.fillStyle = "#000000";
    ctx.fillText(label, labelX + 3, labelY - 2);
  });
}

//Update sidebar detections
function updateDetections(faces) {
  const list = document.getElementById("detections-list");
  list.innerHTML = "";

  if (faces.length === 0) {
    list.innerHTML = '<div class="no-detection">No faces detected</div>';
    document.getElementById("stat-faces").textContent = "0";
    return;
  }

  document.getElementById("stat-faces").textContent = faces.length;

  faces.forEach(face => {
    const isUnknown = face.name === "Unknown";
    const isLow = face.confidence < threshold + 0.05;

    const item = document.createElement("div");
    item.className = "detection-item" + (isUnknown ? " unknown" : "");
    item.innerHTML = `
      <span class="det-name ${isUnknown ? "unknown" : ""}">${face.name}</span>
      <span class="det-conf ${isLow ? "low" : ""}">${(face.confidence * 100).toFixed(1)}%</span>
    `;
    list.appendChild(item);
  });
}

//Capture & send frame
async function captureAndSend() {
  if (!running || video.readyState < 2) return;

  syncCanvas();

  // Draw current video frame to a temp canvas to get base64
  const tmp = document.createElement("canvas");
  tmp.width = video.videoWidth;
  tmp.height = video.videoHeight;
  tmp.getContext("2d").drawImage(video, 0, 0);
  const dataUrl = tmp.toDataURL("image/jpeg", 0.85);

  const t0 = performance.now();
  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl })
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();
    const latency = Math.round(performance.now() - t0);

    // Filter by current threshold (server uses 0.7 default, but we re-filter client-side)
    const filtered = data.faces.map(f => ({
      ...f,
      name: f.confidence >= threshold ? f.name : "Unknown"
    }));

    drawFaces(filtered);
    updateDetections(filtered);

    // Stats
    document.getElementById("stat-latency").textContent = latency;
    totalDetected += filtered.filter(f => f.name !== "Unknown").length;
    document.getElementById("stat-total").textContent = totalDetected;

    // FPS
    frameCount++;
    const now = Date.now();
    if (now - fpsTimer >= 1000) {
      currentFps = frameCount;
      frameCount = 0;
      fpsTimer = now;
      document.getElementById("stat-fps").textContent = currentFps;
      const pct = Math.min(100, (currentFps / MAX_FPS_BAR) * 100);
      document.getElementById("fps-bar").style.width = pct + "%";
    }

    if (filtered.length > 0) {
      const names = filtered.map(f => f.name).join(", ");
      log(`Detected: ${names}`, "ok");
    }

  } catch (err) {
    log(`Error: ${err.message}`, "err");
    setStatus("error");
  }
}

//Start
btnStart.addEventListener("click", async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    video.srcObject = stream;
    await video.play();

    placeholder.style.display = "none";
    running = true;
    scanLine.classList.add("active");
    setStatus("live");
    btnStart.disabled = true;
    btnStop.disabled = false;
    log("Camera started", "ok");

    intervalId = setInterval(captureAndSend, CAPTURE_INTERVAL_MS);
  } catch (err) {
    log(`Camera error: ${err.message}`, "err");
    setStatus("error");
  }
});

//Stop
btnStop.addEventListener("click", () => {
  running = false;
  clearInterval(intervalId);
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null;
  video.srcObject = null;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  placeholder.style.display = "flex";
  scanLine.classList.remove("active");
  setStatus("");
  btnStart.disabled = false;
  btnStop.disabled = true;
  updateDetections([]);
  log("Camera stopped");
  document.getElementById("stat-fps").textContent = "--";
  document.getElementById("stat-latency").textContent = "--";
});