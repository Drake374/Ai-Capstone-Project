const numaanForm = document.getElementById("numaanForm");
const numaanStudentId = document.getElementById("numaanStudentId");
const numaanVideo = document.getElementById("numaanVideo");
const numaanBtn = document.getElementById("numaanBtn");
const numaanStatus = document.getElementById("numaanStatus");

const numaanPreviewWrap = document.getElementById("numaanPreviewWrap");
const numaanPreview = document.getElementById("numaanPreview");

const numaanResult = document.getElementById("numaanResult");
const numaanBadge = document.getElementById("numaanBadge");
const numaanPresent = document.getElementById("numaanPresent");
const numaanSim = document.getElementById("numaanSim");
const numaanFrames = document.getElementById("numaanFrames");
const numaanUsable = document.getElementById("numaanUsable");
const numaanRaw = document.getElementById("numaanRaw");
const numaanExportBtn = document.getElementById("numaanExportBtn");


function setStatus(msg, type = "info") {
  numaanStatus.classList.remove("hidden");
  const colors = {
    info: "text-slate-300",
    ok: "text-green-400",
    err: "text-red-400",
  };
  numaanStatus.className = `text-sm ${colors[type] ?? colors.info}`;
  numaanStatus.textContent = msg;
}

function setLoading(isLoading) {
  numaanBtn.disabled = isLoading;
  numaanBtn.textContent = isLoading ? "Processing..." : "Mark Attendance";
  numaanBtn.classList.toggle("opacity-70", isLoading);
  numaanBtn.classList.toggle("cursor-not-allowed", isLoading);
}

numaanVideo.addEventListener("change", () => {
  const file = numaanVideo.files?.[0];
  if (!file) return;

  const url = URL.createObjectURL(file);
  numaanPreview.src = url;
  numaanPreviewWrap.classList.remove("hidden");
});

numaanForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const id = numaanStudentId.value.trim();
  const file = numaanVideo.files?.[0];

  if (!id) return setStatus("Please enter Student ID.", "err");
  if (!file) return setStatus("Please choose a video file.", "err");

  setStatus("Uploading video and running verification…", "info");
  setLoading(true);

  try {
    const fd = new FormData();
    fd.append("student_id", id);
    fd.append("video", file);

    const res = await fetch("/attendance", { method: "POST", body: fd });
    const data = await res.json();

    numaanResult.classList.remove("hidden");
    numaanRaw.textContent = JSON.stringify(data, null, 2);

    const isPresent = Boolean(data.present);
    numaanPresent.textContent = isPresent ? "YES ✅" : "NO ❌";
    numaanSim.textContent = (data.similarity_median ?? "-").toString();
    numaanFrames.textContent = (data.frames_extracted ?? "-").toString();
    numaanUsable.textContent = (data.usable_face_frames ?? "-").toString();

    if (isPresent) {
      numaanBadge.textContent = "PRESENT";
      numaanBadge.className = "text-xs px-3 py-1 rounded-full bg-green-600/20 text-green-300 border border-green-700";
      setStatus("Attendance marked: PRESENT ✅", "ok");
    } else {
      numaanBadge.textContent = "ABSENT";
      numaanBadge.className = "text-xs px-3 py-1 rounded-full bg-red-600/20 text-red-300 border border-red-700";
      setStatus(`Attendance marked: ABSENT ❌ (reason: ${data.reason ?? "unknown"})`, "err");
    }
  } catch (err) {
    console.error(err);
    setStatus("Request failed. Check server logs / network.", "err");
  } finally {
    setLoading(false);
  }

  
});

numaanExportBtn.addEventListener("click", async () => {
  try {
    // We'll fetch to detect "no file yet" JSON errors.
    const res = await fetch("/export");

    const ct = res.headers.get("content-type") || "";
    if (ct.includes("application/json")) {
      const data = await res.json();
      alert(data.error || "No CSV available yet.");
      return;
    }

    // If it's a CSV file, download it
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "attendance_log.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();

    URL.revokeObjectURL(url);
  } catch (e) {
    console.error(e);
    alert("Export failed. Is the server running?");
  }
});

