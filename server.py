import os
import uuid
import json
import asyncio
import threading
import shutil
from typing import Dict
import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine.video_processor import get_video_metadata, extract_frame_at_time, VideoProcessingTask
from engine.mask_generator import generate_mask
from engine.inpainter import (
    TeleaInpainter, NavierStokesInpainter, BlurInpainter, LamaInpainter,
    LAMA_MODEL_PATH, download_lama_model
)

app = FastAPI(title="Gemini AI Video Watermark Remover")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

for d in [STATIC_DIR, UPLOADS_DIR, OUTPUTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-memory stores
video_registry: Dict[str, dict] = {}
tasks: Dict[str, VideoProcessingTask] = {}
model_download_status = {
    "is_downloading": False,
    "progress": 0.0,
    "error": None
}

class ProcessRequest(BaseModel):
    video_id: str
    method: str = "telea"  # 'lama', 'telea', 'navier_stokes', 'delogo', 'blur'
    x: int
    y: int
    w: int
    h: int
    shape: str = "star"
    dilation: int = 3
    feather: int = 0

@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

ALLOWED_VIDEO_EXTS = [".mp4", ".mov", ".mkv", ".webm", ".avi"]
ALLOWED_IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
ALLOWED_EXTS = ALLOWED_VIDEO_EXTS + ALLOWED_IMAGE_EXTS

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported format. Please upload MP4, MOV, MKV, WebM video or PNG, JPG, WebP image."
        )
    
    video_id = str(uuid.uuid4())[:8]
    save_filename = f"{video_id}{ext}"
    save_path = os.path.join(UPLOADS_DIR, save_filename)
    
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        meta = get_video_metadata(save_path)
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=400, detail=f"Failed to inspect media: {str(e)}")
        
    video_registry[video_id] = {
        "id": video_id,
        "filename": file.filename,
        "path": save_path,
        "meta": meta
    }
    
    return {
        "video_id": video_id,
        "filename": file.filename,
        "meta": meta
    }

@app.get("/api/video/{video_id}")
async def serve_video(video_id: str):
    if video_id in video_registry:
        path = video_registry[video_id]["path"]
    elif video_id in tasks:
        path = tasks[video_id].output_path
    else:
        raise HTTPException(status_code=404, detail="Media not found")

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    ext = os.path.splitext(path)[1].lower()
    if ext in ALLOWED_IMAGE_EXTS:
        media_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else f"image/{ext.lstrip('.')}"
        return FileResponse(path, media_type=media_type)
    return FileResponse(path, media_type="video/mp4")

@app.get("/api/preview-frame")
async def preview_frame(
    video_id: str = Query(...),
    time_sec: float = Query(0.0),
    mode: str = Query("mask"),  # 'original', 'mask', 'inpaint'
    method: str = Query("telea"),
    x: int = Query(...),
    y: int = Query(...),
    w: int = Query(...),
    h: int = Query(...),
    shape: str = Query("star"),
    dilation: int = Query(3),
    feather: int = Query(0)
):
    if video_id not in video_registry:
        raise HTTPException(status_code=404, detail="Video not found")
        
    path = video_registry[video_id]["path"]
    frame = extract_frame_at_time(path, time_sec)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not extract frame at given timestamp")
        
    height, width = frame.shape[:2]
    bbox = {"x": x, "y": y, "w": w, "h": h}
    mask = generate_mask(height, width, x, y, w, h, shape=shape, dilation=dilation, feather=feather)

    if mode == "mask":
        # Create visual overlay: semi-transparent cyan/violet for mask and red bounding box
        overlay = frame.copy()
        # Tint masked pixels cyan
        overlay[mask > 0] = [255, 120, 30]  # BGR
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
    elif mode == "inpaint":
        if method == "lama":
            inpainter = LamaInpainter()
        elif method == "navier_stokes":
            inpainter = NavierStokesInpainter()
        elif method == "blur":
            inpainter = BlurInpainter()
        else:
            inpainter = TeleaInpainter()
        frame = inpainter.inpaint(frame, mask, bbox)

    ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return StreamingResponse(iter([buffer.tobytes()]), media_type="image/jpeg")

@app.post("/api/process")
async def start_process(req: ProcessRequest, background_tasks: BackgroundTasks):
    if req.video_id not in video_registry:
        raise HTTPException(status_code=404, detail="Video not found")

    input_path = video_registry[req.video_id]["path"]
    ext = os.path.splitext(input_path)[1].lower()
    task_id = str(uuid.uuid4())[:8]
    if ext in ALLOWED_IMAGE_EXTS:
        output_filename = f"watermark_removed_{task_id}{ext}"
    else:
        output_filename = f"watermark_removed_{task_id}.mp4"
    output_path = os.path.join(OUTPUTS_DIR, output_filename)

    task = VideoProcessingTask(
        task_id=task_id,
        input_path=input_path,
        output_path=output_path,
        method=req.method,
        bbox={"x": req.x, "y": req.y, "w": req.w, "h": req.h},
        shape=req.shape,
        dilation=req.dilation,
        feather=req.feather
    )

    tasks[task_id] = task
    # Launch in worker thread
    threading.Thread(target=task.run, daemon=True).start()

    return {"task_id": task_id, "status": "started"}

@app.get("/api/task-status/{task_id}")
async def task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
        
    t = tasks[task_id]
    ext = os.path.splitext(t.output_path)[1].lower()
    is_image = ext in ALLOWED_IMAGE_EXTS
    return {
        "task_id": t.task_id,
        "is_image": is_image,
        "status": t.status,
        "progress": t.progress,
        "current_frame": t.current_frame,
        "total_frames": t.total_frames,
        "fps": t.fps,
        "eta_seconds": t.eta_seconds,
        "error_message": t.error_message,
        "has_output": os.path.exists(t.output_path) if t.status == "completed" else False
    }

@app.post("/api/cancel/{task_id}")
async def cancel_task(task_id: str):
    if task_id in tasks:
        tasks[task_id].cancel()
        return {"status": "cancelled"}
    raise HTTPException(status_code=404, detail="Task not found")

@app.get("/api/download/{task_id}")
async def download_output(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
        
    path = tasks[task_id].output_path
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Output file not ready")
        
    ext = os.path.splitext(path)[1].lower()
    if ext in ALLOWED_IMAGE_EXTS:
        media_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else f"image/{ext.lstrip('.')}"
        filename = f"clean_gemini_image_{task_id}{ext}"
    else:
        media_type = "video/mp4"
        filename = f"clean_gemini_video_{task_id}.mp4"

    return FileResponse(
        path,
        media_type=media_type,
        filename=filename
    )

@app.get("/api/model-status")
async def model_status():
    has_lama = os.path.exists(LAMA_MODEL_PATH) and os.path.getsize(LAMA_MODEL_PATH) > 100_000_000
    return {
        "has_lama": has_lama,
        "is_downloading": model_download_status["is_downloading"],
        "progress": model_download_status["progress"],
        "error": model_download_status["error"]
    }

@app.post("/api/download-lama-model")
async def trigger_lama_download(background_tasks: BackgroundTasks):
    if model_download_status["is_downloading"]:
        return {"status": "already_downloading"}

    def worker():
        model_download_status["is_downloading"] = True
        model_download_status["error"] = None
        model_download_status["progress"] = 0.0

        def callback(downloaded, total, pct):
            model_download_status["progress"] = round(pct, 1)

        try:
            download_lama_model(callback)
        except Exception as e:
            model_download_status["error"] = str(e)
        finally:
            model_download_status["is_downloading"] = False

    threading.Thread(target=worker, daemon=True).start()
    return {"status": "started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
