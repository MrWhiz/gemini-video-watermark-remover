# 🎬 Gemini AI Video & Image Watermark Remover

A high-performance, 100% local, web-based tool designed to cleanly remove the Google Gemini AI watermark (the 4-pointed translucent sparkle star) and other watermarks from both **videos** and **images**.

It combines **CUDA GPU-accelerated neural inpainting (LaMa)**, **OpenCV boundary propagation (Telea / Navier-Stokes)**, and **FFmpeg stream filters** with intuitive click-and-drag mask positioning, interactive frame scrubbing, precise mask tuning, and lossless audio preservation.

---

## ✨ Features

- **Full Video & Image Support**:
  - Remove watermarks from AI-generated videos (`MP4`, `MOV`, `MKV`, `WebM`) and images (`PNG`, `JPG`, `JPEG`, `WebP`, `BMP`).
  - Near-instantaneous inpainting (<0.2s) for images with full-resolution export.
- **Interactive Click & Draggable Mask Targeting**:
  - Simply **click and drag** anywhere directly on the preview frame to fluidly move and position the mask over the watermark in real time.
  - Live visual bounding box provides immediate sub-millisecond feedback while dragging.
  - Proportional bidirectional scaling keeps Gemini Star and Circle contours perfectly square when adjusting size.
- **Multi-Method Inpainting Engine**:
  - **LaMa AI (Neural Inpainting)**: Deep learning texture synthesis with Fast Fourier Convolutions. Hardware-accelerated with **NVIDIA CUDA GPU** (e.g. RTX 4060 Ti) running at ~10 FPS for video, and <0.2s for images.
  - **OpenCV Telea (Fast Marching)**: Propagates surrounding pixel gradients into the star shape. Clean edges and runs at 60+ FPS without ML overhead.
  - **OpenCV Navier-Stokes**: Fluid dynamics PDE inpainting, ideal for continuous gradients and subtle textures.
  - **FFmpeg Delogo**: Ultra-fast direct stream filter for rapid batch jobs.
  - **Gaussian Conceal**: Soft feathered blur over the watermark area.
- **Contour-Matched Masking**:
  - Generates an exact **4-pointed astroid star contour** matching the Gemini icon geometry so only the watermark pixels are touched while surrounding background remains 100% untouched.
  - Supports rectangular box and circle contours.
- **Interactive Web Studio**:
  - Drag-and-drop video and image uploader.
  - Timeline scrubber to preview any frame (automatically disabled for static images).
  - **"Test Inpaint On Frame"** toggle: View instantaneous before/after inpainting on a single frame in <100ms before processing full files.
- **Lossless Audio Preservation**:
  - Automatically copies and re-muxes original audio tracks (`-c:a copy`) for videos without re-encoding or audio sync loss.
- **100% Private & Local**:
  - No cloud uploads, no API keys, and no rate limits. All processing runs entirely on your local machine.

---

## 📋 Prerequisites

1. **Python**: Python 3.10 to 3.13 installed.
2. **FFmpeg**: FFmpeg installed and accessible in your system's `PATH` (for video rendering and audio muxing).
   - To verify: Open a terminal and run `ffmpeg -version`.
3. **GPU (Recommended)**:
   - NVIDIA GPU with CUDA 12 support (e.g., RTX 30/40 series) for high-speed AI inpainting via `onnxruntime-gpu`.
   - CPU mode is supported automatically as fallback.

---

## 🚀 Quick Start (Windows)

The simplest way to run the application is to use the included launcher script:

1. Double-click **`start.bat`** in the project directory.
2. Open your browser and go to:
   ```
   http://127.0.0.1:8000
   ```

---

## 🛠️ Manual Installation & Setup

If setting up manually from scratch:

### 1. Open Terminal in Project Directory
```powershell
cd F:\AntiGravity\VideoEditor
```

### 2. Create and Activate Virtual Environment
```powershell
# Create venv
python -m venv venv

# Activate venv on Windows PowerShell
.\venv\Scripts\Activate.ps1

# Or on Command Prompt (CMD)
.\venv\Scripts\activate.bat
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Start the Application Server
```powershell
python server.py
```

Open **`http://localhost:8000`** in any modern web browser.

---

## 📖 Step-by-Step Usage Guide

### Step 1: Upload Video or Image
- Drag and drop your video or image file into the upload zone, or click to browse.
- Supported video formats: `.mp4`, `.mov`, `.mkv`, `.webm`.
- Supported image formats: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`.
- Media metadata (resolution, frame rate, duration, audio status) will be displayed.

### Step 2: Position the Mask (Click & Drag)
- Simply **click and drag** anywhere directly over the watermark on the preview frame to position the mask.
- Fine-tune using the sliders:
  - **Horizontal (X) / Vertical (Y)**: Fine coordinate adjustments.
  - **Width (W) / Height (H)**: Scale bounding box size (automatically kept square for Star and Circle).
  - **Mask Contour**: Select **✦ Gemini Star** (recommended), **▭ Rectangle Box**, or **◯ Circle**.
  - **Mask Expansion (Dilation)**: Expands the mask by a few pixels to cover anti-aliased translucent edges.
  - **Soft Feather Border**: Softens the mask transition boundary.

### Step 3: Test on a Single Frame
- For videos, scrub the timeline slider to any frame.
- Click **"Test Inpaint On Frame"** to see an immediate live inpaint preview on that frame.
- Switch between **"Show Mask Overlay"**, **"Test Inpaint On Frame"**, and **"Original Frame"** to verify quality.

### Step 4: Choose Inpainting Method
- **AI Inpainting (LaMa)** *(Recommended for photorealistic texture)*:
  - Deep neural inpainting running accelerated on NVIDIA CUDA GPU.
  - Downloads the model weights once automatically (~200MB).
- **OpenCV Telea** *(Recommended for speed)*: Fast, crisp, zero ML download required.
- **Navier-Stokes**: Smooth gradient propagation.
- **FFmpeg Delogo**: Instant batch stream removal (video only).
- **Gaussian Conceal**: Soft blur.

### Step 5: Export Video / Image
- Click **"Remove Watermark & Export"**.
- For images, the clean output is produced almost instantly (<0.2s).
- For videos, watch live progress: frame count, processing speed (FPS), and estimated time remaining (ETA).
- Once completed, preview the final clean media in the player and click **"Download Clean Media"**.

---

## 📂 Project Structure

```text
VideoEditor/
├── engine/
│   ├── mask_generator.py      # Astroid star contour & mask geometry generator
│   ├── inpainter.py           # Inpainting algorithms (LaMa CUDA, Telea, NS, Blur)
│   └── video_processor.py     # FFmpeg video pipeline, image handler & audio muxing
├── models/                    # Cached ONNX model weights (auto-downloaded)
├── static/
│   ├── index.html             # Studio user interface (video & image)
│   ├── style.css              # Dark mode styling & glassmorphism design system
│   └── app.js                 # Frontend application controller with drag-to-target
├── uploads/                   # Staged input media files
├── outputs/                   # Exported watermark-removed videos and images
├── server.py                  # FastAPI server & REST API
├── requirements.txt           # Python package requirements
├── start.bat                  # One-click Windows startup script
└── README.md                  # Project documentation
```

---

## 📄 License

MIT License. Free for personal and commercial use.
