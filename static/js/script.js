// ============================================================
// Face Mask Detection — frontend logic
// ============================================================

// ---- DOM ----
const video        = document.getElementById('video');
const uploadPreview= document.getElementById('uploadPreview');
const overlay      = document.getElementById('overlay');
const placeholder  = document.getElementById('placeholder');
const statusPill   = document.getElementById('statusPill');

const tabLive      = document.getElementById('tab-live');
const tabUpload    = document.getElementById('tab-upload');
const liveControls = document.getElementById('liveControls');
const uploadControls = document.getElementById('uploadControls');

const startBtn  = document.getElementById('startBtn');
const stopBtn   = document.getElementById('stopBtn');
const switchBtn = document.getElementById('switchBtn');
const fileInput = document.getElementById('fileInput');
const clearBtn  = document.getElementById('clearBtn');

const verdict        = document.getElementById('verdict');
const verdictIcon    = verdict.querySelector('.verdict-icon');
const verdictText    = verdict.querySelector('.verdict-text');
const confValue      = document.getElementById('confValue');
const meterFill      = document.getElementById('meterFill');
const recommendation = document.getElementById('recommendation');
const fpsValue       = document.getElementById('fpsValue');
const countValue     = document.getElementById('countValue');

// ---- State ----
let stream = null;
let loop = null;
let facingMode = 'user';
let busy = false;
let checks = 0;
const ctx = overlay.getContext('2d');

// ---- Label config (matches model classes: mask / no_mask / improper_mask) ----
const LABELS = {
    mask:          { kind: 'ok',   text: 'Mask On',        color: '#2ec27e', icon: '😷', tip: '✅ Great — mask worn correctly.' },
    with_mask:     { kind: 'ok',   text: 'Mask On',        color: '#2ec27e', icon: '😷', tip: '✅ Great — mask worn correctly.' },
    improper_mask: { kind: 'warn', text: 'Improper Mask',  color: '#f5b945', icon: '⚠️', tip: '⚠️ Adjust your mask to cover nose and mouth.' },
    incorrect_mask:{ kind: 'warn', text: 'Improper Mask',  color: '#f5b945', icon: '⚠️', tip: '⚠️ Adjust your mask to cover nose and mouth.' },
    no_mask:       { kind: 'bad',  text: 'No Mask',        color: '#ef4d56', icon: '🚫', tip: '❌ No mask detected. Please wear a mask.' },
    without_mask:  { kind: 'bad',  text: 'No Mask',        color: '#ef4d56', icon: '🚫', tip: '❌ No mask detected. Please wear a mask.' },
};
const NEUTRAL = { kind: 'neutral', text: 'No face', color: '#9aa7b8', icon: '🙂', tip: 'No face detected in frame.' };

// ============================================================
// Tabs
// ============================================================
tabLive.addEventListener('click', () => switchTab('live'));
tabUpload.addEventListener('click', () => switchTab('upload'));

function switchTab(mode) {
    const isLive = mode === 'live';
    tabLive.classList.toggle('active', isLive);
    tabUpload.classList.toggle('active', !isLive);
    liveControls.hidden = !isLive;
    uploadControls.hidden = isLive;

    stopCamera();
    clearOverlay();
    resetResult();
    uploadPreview.hidden = true;
    video.hidden = false;
    placeholder.style.display = 'flex';
    placeholder.querySelector('p').textContent = isLive ? 'Camera is off' : 'No image yet';
    placeholder.querySelector('span').innerHTML = isLive
        ? 'Press <strong>Start Camera</strong> to begin detection'
        : 'Press <strong>Choose Image</strong> to run detection';
}

// ============================================================
// Live camera
// ============================================================
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);
switchBtn.addEventListener('click', async () => {
    facingMode = facingMode === 'user' ? 'environment' : 'user';
    if (stream) { stopCamera(); await startCamera(); }
});

async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
        setStatus('error', 'Camera not supported');
        recommendation.textContent = 'Your browser does not support camera access. Try Chrome, Edge or Firefox.';
        return;
    }
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode }
        });
        video.srcObject = stream;
        await video.play();

        placeholder.style.display = 'none';
        setStatus('live', '● Live');
        startBtn.disabled = true;
        stopBtn.disabled = false;
        switchBtn.disabled = false;

        clearInterval(loop);
        loop = setInterval(runLiveDetection, 500);
    } catch (err) {
        setStatus('error', 'Camera blocked');
        recommendation.textContent = `Could not access camera: ${err.name}. Please allow camera permission.`;
    }
}

function stopCamera() {
    clearInterval(loop); loop = null;
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    video.srcObject = null;
    setStatus('off', '● Idle');
    startBtn.disabled = false;
    stopBtn.disabled = true;
    switchBtn.disabled = true;
    placeholder.style.display = 'flex';
    clearOverlay();
}

async function runLiveDetection() {
    if (busy || video.readyState < 2) return;
    busy = true;
    try {
        const w = video.videoWidth, h = video.videoHeight;
        const cap = document.createElement('canvas');
        cap.width = w; cap.height = h;
        cap.getContext('2d').drawImage(video, 0, 0, w, h);
        const data = await detect(cap.toDataURL('image/jpeg', 0.7));
        render(data, w, h);
    } catch (e) {
        console.error(e);
    } finally {
        busy = false;
    }
}

// ============================================================
// Upload mode
// ============================================================
fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async () => {
        video.hidden = true;
        uploadPreview.hidden = false;
        uploadPreview.src = reader.result;
        placeholder.style.display = 'none';
        clearBtn.disabled = false;
        await uploadPreview.decode().catch(() => {});
        setStatus('live', '● Analyzing');
        const data = await detect(reader.result);
        render(data, uploadPreview.naturalWidth, uploadPreview.naturalHeight);
        setStatus('off', '● Done');
    };
    reader.readAsDataURL(file);
});

clearBtn.addEventListener('click', () => {
    uploadPreview.hidden = true; uploadPreview.src = '';
    video.hidden = false;
    fileInput.value = '';
    clearBtn.disabled = true;
    placeholder.style.display = 'flex';
    clearOverlay(); resetResult(); setStatus('off', '● Idle');
});

// ============================================================
// Backend call
// ============================================================
async function detect(imageData) {
    const res = await fetch('/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData })
    });
    return res.json();
}

// ============================================================
// Rendering
// ============================================================
function render(data, srcW, srcH) {
    if (!data || data.status !== 'success') {
        setStatus('error', 'Error');
        recommendation.textContent = data?.message || 'Detection failed.';
        return;
    }

    checks++;
    const dets = data.detections || [];
    countValue.textContent = dets.length;

    drawBoxes(dets, srcW, srcH);

    if (!dets.length) { applyVerdict(NEUTRAL, 0); return; }

    const best = dets.reduce((a, b) => (b.confidence > a.confidence ? b : a));
    const cfg = LABELS[best.label] || NEUTRAL;
    applyVerdict(cfg, best.confidence);
}

function applyVerdict(cfg, confidence) {
    verdict.className = `verdict ${cfg.kind}`;
    verdictIcon.textContent = cfg.icon;
    verdictText.textContent = cfg.text;
    recommendation.textContent = cfg.tip;

    const pct = Math.round(confidence * 100);
    confValue.textContent = `${pct}%`;
    meterFill.style.width = `${pct}%`;
    meterFill.style.background = cfg.kind === 'neutral'
        ? 'linear-gradient(90deg,#4f8cff,#7c5cff)'
        : cfg.color;
}

function drawBoxes(dets, srcW, srcH) {
    overlay.width = srcW;
    overlay.height = srcH;
    ctx.clearRect(0, 0, srcW, srcH);
    ctx.lineWidth = Math.max(2, srcW / 320);
    ctx.font = `${Math.max(14, srcW / 40)}px Inter, sans-serif`;

    for (const d of dets) {
        const cfg = LABELS[d.label] || NEUTRAL;
        const [x1, y1, x2, y2] = d.box;
        ctx.strokeStyle = cfg.color;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        const tag = `${cfg.text} ${(d.confidence * 100).toFixed(0)}%`;
        const tw = ctx.measureText(tag).width + 12;
        const th = Math.max(20, srcW / 24);
        ctx.fillStyle = cfg.color;
        ctx.fillRect(x1, Math.max(0, y1 - th), tw, th);
        ctx.fillStyle = '#0d1117';
        ctx.fillText(tag, x1 + 6, Math.max(th - 6, y1 - 6));
    }
}

function clearOverlay() { ctx.clearRect(0, 0, overlay.width, overlay.height); }

function resetResult() {
    verdict.className = 'verdict neutral';
    verdictIcon.textContent = '⏳';
    verdictText.textContent = 'Waiting…';
    confValue.textContent = '0%';
    meterFill.style.width = '0%';
    countValue.textContent = '0';
    recommendation.textContent = 'Start the camera or upload an image to check mask compliance.';
}

function setStatus(kind, text) {
    statusPill.className = `status-pill ${kind}`;
    statusPill.textContent = text;
}

// ---- simple checks/sec counter ----
setInterval(() => { fpsValue.textContent = checks; checks = 0; }, 1000);

window.addEventListener('beforeunload', stopCamera);
