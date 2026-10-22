// Gemini Watermark Remover Web App Controller

let currentVideoId = null;
let videoMetadata = null;
let currentTaskId = null;
let progressPollTimer = null;
let debounceTimer = null;
let activeMode = 'mask'; // 'mask', 'inpaint', 'original'
let selectedMethod = 'telea';
let selectedShape = 'star';

// Elements
const dropzone = document.getElementById('dropzone');
const videoFileInput = document.getElementById('video-file-input');
const uploadPanel = document.getElementById('upload-panel');
const editorWorkspace = document.getElementById('editor-workspace');
const videoMetaBar = document.getElementById('video-meta-bar');

const previewImg = document.getElementById('preview-image');
const previewLoader = document.getElementById('preview-loader');
const previewWrapper = document.getElementById('preview-wrapper');
const timelineSlider = document.getElementById('timeline-slider');
const timelineReadout = document.getElementById('timeline-readout');

const sliderX = document.getElementById('slider-x');
const sliderY = document.getElementById('slider-y');
const sliderW = document.getElementById('slider-w');
const sliderH = document.getElementById('slider-h');
const sliderDilation = document.getElementById('slider-dilation');
const sliderFeather = document.getElementById('slider-feather');

const valX = document.getElementById('val-x');
const valY = document.getElementById('val-y');
const valW = document.getElementById('val-w');
const valH = document.getElementById('val-h');
const valDilation = document.getElementById('val-dilation');
const valFeather = document.getElementById('val-feather');

const dragIndicator = document.getElementById('drag-indicator-box');
let isDraggingMask = false;
const shapeBtns = document.querySelectorAll('.shape-btn');
const modeBtns = document.querySelectorAll('.btn-toggle');
const methodCards = document.querySelectorAll('.method-card');
const btnStartProcess = document.getElementById('btn-start-process');

const progressCard = document.getElementById('progress-card');
const progressStatusLabel = document.getElementById('progress-status-label');
const progressPercentage = document.getElementById('progress-percentage');
const progressBarFill = document.getElementById('progress-bar-fill');
const statFrames = document.getElementById('stat-frames');
const statFps = document.getElementById('stat-fps');
const statEta = document.getElementById('stat-eta');
const statMethod = document.getElementById('stat-method');
const btnCancelTask = document.getElementById('btn-cancel-task');

const resultCard = document.getElementById('result-card');
const resultVideoPlayer = document.getElementById('result-video-player');
const btnDownloadVideo = document.getElementById('btn-download-video');
const btnReprocess = document.getElementById('btn-reprocess');

// --- Initialization & Event Listeners ---

document.addEventListener('DOMContentLoaded', () => {
  setupUploadEvents();
  setupSliderEvents();
  setupPreviewModeEvents();
  setupShapeEvents();
  setupMethodEvents();
  setupProcessingEvents();
  checkModelStatus();
});

// Check LaMa model status
async function checkModelStatus() {
  try {
    const res = await fetch('/api/model-status');
    const data = await res.json();
    const lamaCard = document.getElementById('card-method-lama');
    const statusText = document.getElementById('lama-status-text');
    if (data.has_lama) {
      statusText.textContent = "Ready (NVIDIA CUDA GPU)";
      statusText.style.color = "#34d399";
    } else if (data.is_downloading) {
      statusText.textContent = `Downloading ${data.progress}%...`;
      statusText.style.color = "#fbbf24";
      setTimeout(checkModelStatus, 1500);
    } else {
      statusText.textContent = "Click to Auto-Download (~200MB)";
      statusText.style.color = "#818cf8";
    }
  } catch (e) {
    console.error("Failed to check model status", e);
  }
}

// Upload Handling
function setupUploadEvents() {
  dropzone.addEventListener('click', () => videoFileInput.click());
  
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  videoFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  });
}

async function handleFileUpload(file) {
  const formData = new FormData();
  formData.append('file', file);

  dropzone.innerHTML = `
    <div class="drop-icon">
      <div class="spinner" style="width: 28px; height: 28px;"></div>
    </div>
    <div class="drop-text-main">Uploading & analyzing ${file.name}...</div>
    <div class="drop-text-sub">Please wait a moment while frames and audio are inspected</div>
  `;

  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Upload failed");
    }

    const data = await res.json();
    currentVideoId = data.video_id;
    videoMetadata = data.meta;

    const isImage = data.meta.is_image;
    const btnProcessLabel = document.getElementById('btn-process-label');
    if (btnProcessLabel) {
      btnProcessLabel.textContent = isImage ? "Remove Watermark & Export Image" : "Remove Watermark & Export Video";
    }

    const scrubberRow = document.querySelector('.scrubber-control-row');
    if (scrubberRow) {
      scrubberRow.style.display = isImage ? 'none' : 'flex';
    }

    const specsHtml = isImage ? `
      <span class="spec-pill" style="border-color: rgba(52, 211, 153, 0.4);">Image</span>
      <span class="spec-pill" style="border-color: rgba(52, 211, 153, 0.4);">${data.meta.width} × ${data.meta.height}</span>
    ` : `
      <span class="spec-pill" style="border-color: rgba(52, 211, 153, 0.4);">${data.meta.width} × ${data.meta.height}</span>
      <span class="spec-pill" style="border-color: rgba(52, 211, 153, 0.4);">${data.meta.duration}s</span>
      <span class="spec-pill" style="border-color: rgba(52, 211, 153, 0.4);">${data.meta.fps} FPS</span>
    `;

    dropzone.innerHTML = `
      <div class="drop-icon" style="color: #34d399;">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
          <polyline points="22 4 12 14.01 9 11.01"></polyline>
        </svg>
      </div>
      <div class="drop-text-main" style="color: #34d399;">${data.filename} is ready</div>
      <div class="drop-text-sub">Click or drop another file anytime to replace this ${isImage ? 'image' : 'video'}</div>
      <div class="file-specs">
        ${specsHtml}
      </div>
    `;

    displayVideoMetadata(data.filename, data.meta);
    initializeSliders(data.meta);
    editorWorkspace.classList.remove('hidden');

    // Portrait vs Landscape aspect ratio adjustment
    if (data.meta.height > data.meta.width) {
      previewWrapper.classList.add('portrait');
    } else {
      previewWrapper.classList.remove('portrait');
    }

    // Scroll to editor
    editorWorkspace.scrollIntoView({ behavior: 'smooth' });

    // Initial frame load
    refreshPreviewFrame();

  } catch (error) {
    alert("Error uploading file: " + error.message);
    resetDropzone();
  }
}

function resetDropzone() {
  dropzone.innerHTML = `
    <div class="drop-icon">
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
      </svg>
    </div>
    <div class="drop-text-main">Drop your video or image here or click to browse</div>
    <div class="drop-text-sub">Supports MP4, MOV, MKV, WebM videos &amp; PNG, JPG, WebP images (up to 4K)</div>
    <div class="file-specs">
      <span class="spec-pill">Video &amp; Image Support</span>
      <span class="spec-pill">Local Processing</span>
      <span class="spec-pill">Hardware Accelerated</span>
    </div>
  `;
}

function displayVideoMetadata(filename, meta) {
  const metaFilename = document.getElementById('meta-filename');
  metaFilename.textContent = filename;
  metaFilename.title = filename;
  document.getElementById('meta-resolution').textContent = `${meta.width} × ${meta.height}`;
  document.getElementById('meta-duration').textContent = meta.is_image ? "Image (1 Frame)" : `${meta.duration}s (${meta.total_frames} frames)`;
  document.getElementById('meta-fps').textContent = meta.is_image ? "Static" : `${meta.fps} FPS`;
  document.getElementById('meta-audio').textContent = meta.is_image ? "N/A" : (meta.has_audio ? "Yes (Preserved)" : "None");
  videoMetaBar.classList.remove('hidden');

  timelineSlider.max = meta.duration || 0;
  timelineSlider.value = 0;
  updateTimelineReadout(0);
}

function initializeSliders(meta) {
  sliderX.max = meta.width;
  sliderY.max = meta.height;
  sliderW.max = Math.min(meta.width, 400);
  sliderH.max = Math.min(meta.height, 400);

  const box = meta.suggested_box;
  sliderX.value = box.x;
  sliderY.value = box.y;
  sliderW.value = box.w;
  sliderH.value = box.h;
  sliderDilation.value = box.dilation || 8;
  sliderFeather.value = box.feather || 0;

  if (box.shape) {
    selectedShape = box.shape;
    shapeBtns.forEach(b => {
      b.classList.toggle('active', b.dataset.shape === box.shape);
    });
  }

  if (selectedShape === 'star' || selectedShape === 'circle') {
    sliderH.value = sliderW.value;
  }

  currentMaskW = parseInt(sliderW.value) || 68;
  currentMaskH = parseInt(sliderH.value) || 68;

  updateSliderValueLabels();
}

function updateSliderValueLabels() {
  valX.textContent = `${sliderX.value}px`;
  valY.textContent = `${sliderY.value}px`;
  valW.textContent = `${sliderW.value}px`;
  valH.textContent = `${sliderH.value}px`;
  valDilation.textContent = `${sliderDilation.value}px`;
  valFeather.textContent = `${sliderFeather.value}px`;
}

// Sliders and Position Events
let currentMaskW = 68;
let currentMaskH = 68;
let sliderLiveIndicatorTimer = null;

function showSliderLiveBox() {
  updateDragIndicator(parseInt(sliderX.value), parseInt(sliderY.value), parseInt(sliderW.value), parseInt(sliderH.value));
  clearTimeout(sliderLiveIndicatorTimer);
  sliderLiveIndicatorTimer = setTimeout(hideDragIndicator, 400);
}

function setupSliderEvents() {
  // Width slider: expands symmetrically from center
  sliderW.addEventListener('input', () => {
    const newW = parseInt(sliderW.value) || 10;
    const diffW = newW - currentMaskW;
    currentMaskW = newW;

    const maxBoundW = videoMetadata ? videoMetadata.width : (parseInt(sliderX.max) || 1920);
    const maxBoundH = videoMetadata ? videoMetadata.height : (parseInt(sliderY.max) || 1080);

    let curX = parseInt(sliderX.value) || 0;
    // Shift top-left X by half the width delta so center stays fixed
    let newX = Math.round(curX - diffW / 2);
    newX = Math.max(0, Math.min(Math.max(0, maxBoundW - newW), newX));
    sliderX.value = newX;

    if (selectedShape === 'star' || selectedShape === 'circle') {
      const diffH = newW - currentMaskH;
      currentMaskH = newW;
      sliderH.value = newW;

      let curY = parseInt(sliderY.value) || 0;
      // Shift top-left Y by half the height delta so center stays fixed
      let newY = Math.round(curY - diffH / 2);
      newY = Math.max(0, Math.min(Math.max(0, maxBoundH - newW), newY));
      sliderY.value = newY;
    }

    updateSliderValueLabels();
    showSliderLiveBox();
    schedulePreviewRefresh();
  });

  // Height slider: expands symmetrically from center
  sliderH.addEventListener('input', () => {
    const newH = parseInt(sliderH.value) || 10;
    const diffH = newH - currentMaskH;
    currentMaskH = newH;

    const maxBoundW = videoMetadata ? videoMetadata.width : (parseInt(sliderX.max) || 1920);
    const maxBoundH = videoMetadata ? videoMetadata.height : (parseInt(sliderY.max) || 1080);

    let curY = parseInt(sliderY.value) || 0;
    // Shift top-left Y by half the height delta so center stays fixed
    let newY = Math.round(curY - diffH / 2);
    newY = Math.max(0, Math.min(Math.max(0, maxBoundH - newH), newY));
    sliderY.value = newY;

    if (selectedShape === 'star' || selectedShape === 'circle') {
      const diffW = newH - currentMaskW;
      currentMaskW = newH;
      sliderW.value = newH;

      let curX = parseInt(sliderX.value) || 0;
      // Shift top-left X by half the width delta so center stays fixed
      let newX = Math.round(curX - diffW / 2);
      newX = Math.max(0, Math.min(Math.max(0, maxBoundW - newH), newX));
      sliderX.value = newX;
    }

    updateSliderValueLabels();
    showSliderLiveBox();
    schedulePreviewRefresh();
  });

  // X and Y position sliders
  [sliderX, sliderY].forEach(slider => {
    slider.addEventListener('input', () => {
      updateSliderValueLabels();
      showSliderLiveBox();
      schedulePreviewRefresh();
    });
  });

  // Dilation and Feather sliders
  [sliderDilation, sliderFeather].forEach(slider => {
    slider.addEventListener('input', () => {
      updateSliderValueLabels();
      schedulePreviewRefresh();
    });
  });

  timelineSlider.addEventListener('input', () => {
    updateTimelineReadout(parseFloat(timelineSlider.value));
    schedulePreviewRefresh();
  });
}

function updateTimelineReadout(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(1);
  timelineReadout.textContent = `${mins.toString().padStart(2, '0')}:${secs.padStart(4, '0')}`;
}

function schedulePreviewRefresh() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(refreshPreviewFrame, 120);
}

// Fetch preview frame
async function refreshPreviewFrame() {
  if (!currentVideoId) return;

  previewLoader.classList.add('active');

  const params = new URLSearchParams({
    video_id: currentVideoId,
    time_sec: timelineSlider.value,
    mode: activeMode,
    method: selectedMethod,
    x: sliderX.value,
    y: sliderY.value,
    w: sliderW.value,
    h: sliderH.value,
    shape: selectedShape,
    dilation: sliderDilation.value,
    feather: sliderFeather.value
  });

  const url = `/api/preview-frame?${params.toString()}`;
  const img = new Image();
  img.onload = () => {
    previewImg.src = url;
    previewLoader.classList.remove('active');
    if (!isDraggingMask) {
      hideDragIndicator();
    }
  };
  img.onerror = () => {
    previewLoader.classList.remove('active');
  };
  img.src = url;
}

// Mode toggle
function getImageRenderedBounds(img, naturalW, naturalH) {
  const rect = img.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return null;
  const imgRatio = naturalW / naturalH;
  const containerRatio = rect.width / rect.height;

  let renderW, renderH, offsetX, offsetY;
  if (containerRatio > imgRatio) {
    renderH = rect.height;
    renderW = renderH * imgRatio;
    offsetX = (rect.width - renderW) / 2;
    offsetY = 0;
  } else {
    renderW = rect.width;
    renderH = renderW / imgRatio;
    offsetX = 0;
    offsetY = (rect.height - renderH) / 2;
  }

  return {
    left: rect.left + offsetX,
    top: rect.top + offsetY,
    width: renderW,
    height: renderH
  };
}

function updateDragIndicator(videoX, videoY, videoW, videoH) {
  if (!videoMetadata || !previewImg || !dragIndicator) return;
  const bounds = getImageRenderedBounds(previewImg, videoMetadata.width, videoMetadata.height);
  if (!bounds) return;

  const wrapperRect = previewWrapper.getBoundingClientRect();
  const scale = bounds.width / videoMetadata.width;

  const screenX = bounds.left - wrapperRect.left + (videoX * scale);
  const screenY = bounds.top - wrapperRect.top + (videoY * scale);
  const screenW = videoW * scale;
  const screenH = videoH * scale;

  dragIndicator.style.left = `${screenX}px`;
  dragIndicator.style.top = `${screenY}px`;
  dragIndicator.style.width = `${screenW}px`;
  dragIndicator.style.height = `${screenH}px`;
  dragIndicator.classList.add('active');
}

function hideDragIndicator() {
  if (dragIndicator) {
    dragIndicator.classList.remove('active');
  }
}

function handlePointerPosition(clientX, clientY) {
  if (!videoMetadata || !previewImg) return;
  const bounds = getImageRenderedBounds(previewImg, videoMetadata.width, videoMetadata.height);
  if (!bounds) return;

  const relX = Math.max(0, Math.min(bounds.width, clientX - bounds.left));
  const relY = Math.max(0, Math.min(bounds.height, clientY - bounds.top));

  const videoScaleX = videoMetadata.width / bounds.width;
  const videoScaleY = videoMetadata.height / bounds.height;

  const clickX = Math.round(relX * videoScaleX);
  const clickY = Math.round(relY * videoScaleY);

  const w = parseInt(sliderW.value) || 68;
  const h = parseInt(sliderH.value) || 68;

  const posX = Math.max(0, Math.min(videoMetadata.width - w, Math.round(clickX - w / 2)));
  const posY = Math.max(0, Math.min(videoMetadata.height - h, Math.round(clickY - h / 2)));

  sliderX.value = posX;
  sliderY.value = posY;
  currentMaskW = w;
  currentMaskH = h;

  updateSliderValueLabels();
  updateDragIndicator(posX, posY, w, h);
  schedulePreviewRefresh();
}

function setupPreviewModeEvents() {
  // Click and drag directly on video frame to position mask
  previewImg.addEventListener('pointerdown', (e) => {
    if (!videoMetadata) return;
    isDraggingMask = true;
    try {
      previewImg.setPointerCapture(e.pointerId);
    } catch (err) {}
    handlePointerPosition(e.clientX, e.clientY);
  });

  window.addEventListener('pointermove', (e) => {
    if (!isDraggingMask) return;
    e.preventDefault();
    handlePointerPosition(e.clientX, e.clientY);
  });

  window.addEventListener('pointerup', (e) => {
    if (isDraggingMask) {
      isDraggingMask = false;
      try {
        previewImg.releasePointerCapture(e.pointerId);
      } catch (err) {}
      setTimeout(hideDragIndicator, 350);
      refreshPreviewFrame();
    }
  });

  window.addEventListener('pointercancel', () => {
    if (isDraggingMask) {
      isDraggingMask = false;
      hideDragIndicator();
    }
  });

  modeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      modeBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeMode = btn.dataset.mode;
      refreshPreviewFrame();
    });
  });
}

// Shape events
function setupShapeEvents() {
  shapeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      shapeBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      selectedShape = btn.dataset.shape;

      if (selectedShape === 'star' || selectedShape === 'circle') {
        const curW = parseInt(sliderW.value) || 68;
        const curH = parseInt(sliderH.value) || 68;
        const size = Math.max(curW, curH);

        const diffW = size - curW;
        const diffH = size - curH;
        currentMaskW = size;
        currentMaskH = size;
        sliderW.value = size;
        sliderH.value = size;

        const maxBoundW = videoMetadata ? videoMetadata.width : (parseInt(sliderX.max) || 1920);
        const maxBoundH = videoMetadata ? videoMetadata.height : (parseInt(sliderY.max) || 1080);

        let curX = parseInt(sliderX.value) || 0;
        let curY = parseInt(sliderY.value) || 0;
        let newX = Math.round(curX - diffW / 2);
        let newY = Math.round(curY - diffH / 2);
        newX = Math.max(0, Math.min(Math.max(0, maxBoundW - size), newX));
        newY = Math.max(0, Math.min(Math.max(0, maxBoundH - size), newY));

        sliderX.value = newX;
        sliderY.value = newY;
        updateSliderValueLabels();
        showSliderLiveBox();
      }

      schedulePreviewRefresh();
    });
  });
}

// Method selection
function setupMethodEvents() {
  methodCards.forEach(card => {
    card.addEventListener('click', async () => {
      methodCards.forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
      selectedMethod = card.dataset.method;

      // If user selected LaMa AI, check if model needs download
      if (selectedMethod === 'lama') {
        const res = await fetch('/api/model-status');
        const data = await res.json();
        if (!data.has_lama && !data.is_downloading) {
          if (confirm("LaMa AI Inpainting model (~200MB) needs to be downloaded once for local neural inpainting. Download now?")) {
            await fetch('/api/download-lama-model', { method: 'POST' });
            checkModelStatus();
          }
        }
      }

      if (activeMode === 'inpaint') {
        refreshPreviewFrame();
      }
    });
  });
}

// Processing & Progress
function setupProcessingEvents() {
  btnStartProcess.addEventListener('click', startVideoProcessing);
  btnCancelTask.addEventListener('click', cancelActiveTask);
  btnReprocess.addEventListener('click', () => {
    resultCard.classList.remove('active');
    editorWorkspace.scrollIntoView({ behavior: 'smooth' });
  });
}

async function startVideoProcessing() {
  if (!currentVideoId) return;

  btnStartProcess.disabled = true;
  progressCard.classList.add('active');
  resultCard.classList.remove('active');
  progressCard.scrollIntoView({ behavior: 'smooth' });

  const payload = {
    video_id: currentVideoId,
    method: selectedMethod,
    x: parseInt(sliderX.value),
    y: parseInt(sliderY.value),
    w: parseInt(sliderW.value),
    h: parseInt(sliderH.value),
    shape: selectedShape,
    dilation: parseInt(sliderDilation.value),
    feather: parseInt(sliderFeather.value)
  };

  statMethod.textContent = selectedMethod.toUpperCase();

  try {
    const res = await fetch('/api/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      throw new Error("Failed to start processing task");
    }

    const data = await res.json();
    currentTaskId = data.task_id;
    startProgressPolling(currentTaskId);

  } catch (error) {
    alert("Error: " + error.message);
    btnStartProcess.disabled = false;
    progressCard.classList.remove('active');
  }
}

function startProgressPolling(taskId) {
  clearInterval(progressPollTimer);
  progressPollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/api/task-status/${taskId}`);
      if (!res.ok) return;

      const data = await res.json();

      // Update progress bar
      progressPercentage.textContent = `${Math.round(data.progress)}%`;
      progressBarFill.style.width = `${data.progress}%`;

      // Update stats
      statFrames.textContent = `${data.current_frame} / ${data.total_frames}`;
      statFps.textContent = `${data.fps} FPS`;
      statEta.textContent = data.eta_seconds > 0 ? `${data.eta_seconds}s remaining` : (data.status === 'completed' ? 'Done' : 'Finalizing...');

      if (data.status === 'processing') {
        progressStatusLabel.textContent = `Inpainting watermark frames (${Math.round(data.progress)}%)...`;
      } else if (data.status === 'completing') {
        progressStatusLabel.textContent = "Muxing audio & finalizing video...";
      } else if (data.status === 'completed') {
        clearInterval(progressPollTimer);
        handleTaskCompleted(taskId, data.is_image);
      } else if (data.status === 'cancelled') {
        clearInterval(progressPollTimer);
        progressStatusLabel.textContent = "Processing cancelled.";
        btnStartProcess.disabled = false;
      } else if (data.status === 'failed') {
        clearInterval(progressPollTimer);
        alert("Processing failed: " + (data.error_message || "Unknown error"));
        btnStartProcess.disabled = false;
        progressCard.classList.remove('active');
      }
    } catch (e) {
      console.error("Poll error", e);
    }
  }, 400);
}

function handleTaskCompleted(taskId, isImage) {
  progressCard.classList.remove('active');
  btnStartProcess.disabled = false;

  resultCard.classList.add('active');

  const resultImagePlayer = document.getElementById('result-image-player');
  const btnDownloadLabel = document.getElementById('btn-download-label');

  if (isImage || (videoMetadata && videoMetadata.is_image)) {
    resultVideoPlayer.style.display = 'none';
    resultImagePlayer.style.display = 'block';
    resultImagePlayer.src = `/api/video/${taskId}?t=${Date.now()}`;
    if (btnDownloadLabel) btnDownloadLabel.textContent = "Download Clean Image";
  } else {
    resultImagePlayer.style.display = 'none';
    resultVideoPlayer.style.display = 'block';
    resultVideoPlayer.src = `/api/video/${taskId}?t=${Date.now()}`;
    resultVideoPlayer.load();
    if (btnDownloadLabel) btnDownloadLabel.textContent = "Download Clean Video";
  }

  btnDownloadVideo.onclick = () => {
    window.location.href = `/api/download/${taskId}`;
  };

  resultCard.scrollIntoView({ behavior: 'smooth' });
}

async function cancelActiveTask() {
  if (!currentTaskId) return;
  try {
    await fetch(`/api/cancel/${currentTaskId}`, { method: 'POST' });
  } catch (e) {
    console.error(e);
  }
}
