import os
import subprocess
import time
import json
import shutil
import cv2
import numpy as np
from typing import Callable, Optional, Dict, Any

from engine.mask_generator import generate_mask, get_default_gemini_bbox
from engine.inpainter import TeleaInpainter, NavierStokesInpainter, BlurInpainter, LamaInpainter

def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Extracts video width, height, duration, fps, frame count, and audio status.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0
    ret, first_frame = cap.read()
    cap.release()

    # Check for audio stream using ffprobe
    has_audio = False
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_type", "-of", "default=nw=1:nk=1", video_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if "audio" in res.stdout:
            has_audio = True
    except Exception:
        pass

    default_box = get_default_gemini_bbox(width, height)

    return {
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "total_frames": total_frames,
        "duration": round(duration, 2),
        "has_audio": has_audio,
        "suggested_box": default_box
    }

def extract_frame_at_time(video_path: str, timestamp_sec: float) -> Optional[np.ndarray]:
    """
    Extracts a single frame at the given timestamp.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target_frame = int(timestamp_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

class VideoProcessingTask:
    def __init__(
        self,
        task_id: str,
        input_path: str,
        output_path: str,
        method: str,
        bbox: dict,
        shape: str = "star",
        dilation: int = 3,
        feather: int = 0
    ):
        self.task_id = task_id
        self.input_path = input_path
        self.output_path = output_path
        self.method = method  # 'lama', 'telea', 'navier_stokes', 'delogo', 'blur'
        self.bbox = bbox
        self.shape = shape
        self.dilation = dilation
        self.feather = feather
        
        self.is_cancelled = False
        self.progress = 0.0
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 0.0
        self.eta_seconds = 0
        self.status = "queued"  # queued, processing, completing, completed, failed, cancelled
        self.error_message = None

    def cancel(self):
        self.is_cancelled = True
        self.status = "cancelled"

    def run(self):
        try:
            self.status = "processing"
            meta = get_video_metadata(self.input_path)
            width = meta["width"]
            height = meta["height"]
            video_fps = meta["fps"]
            self.total_frames = meta["total_frames"]
            has_audio = meta["has_audio"]

            # Method: FFmpeg Delogo direct filter
            if self.method == "delogo":
                self._run_ffmpeg_delogo(width, height, has_audio)
                return

            # Instantiate Inpainter
            if self.method == "lama":
                inpainter = LamaInpainter()
                if not inpainter.is_available():
                    # Fallback to Telea if LaMa is missing
                    inpainter = TeleaInpainter()
            elif self.method == "navier_stokes":
                inpainter = NavierStokesInpainter()
            elif self.method == "blur":
                inpainter = BlurInpainter()
            else:
                inpainter = TeleaInpainter()

            # Pre-generate watermark mask
            mask = generate_mask(
                height=height,
                width=width,
                x=self.bbox["x"],
                y=self.bbox["y"],
                w=self.bbox["w"],
                h=self.bbox["h"],
                shape=self.shape,
                dilation=self.dilation,
                feather=self.feather
            )

            # Temp output video for raw encoded frames
            temp_video_out = self.output_path + ".temp.mp4"
            if os.path.exists(temp_video_out):
                os.remove(temp_video_out)

            # Setup ffmpeg pipe writer
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{width}x{height}",
                "-pix_fmt", "bgr24",
                "-r", str(video_fps),
                "-i", "-",
                "-an",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                temp_video_out
            ]

            process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            cap = cv2.VideoCapture(self.input_path)
            start_time = time.time()
            frame_idx = 0

            while cap.isOpened():
                if self.is_cancelled:
                    cap.release()
                    try:
                        process.stdin.close()
                    except Exception:
                        pass
                    process.terminate()
                    if os.path.exists(temp_video_out):
                        os.remove(temp_video_out)
                    return

                ret, frame = cap.read()
                if not ret:
                    break

                # Apply inpainting
                processed_frame = inpainter.inpaint(frame, mask, self.bbox)

                # Write to ffmpeg pipe
                try:
                    process.stdin.write(processed_frame.tobytes())
                except (BrokenPipeError, OSError):
                    break

                frame_idx += 1
                self.current_frame = frame_idx

                # Compute performance metrics every 5 frames
                if frame_idx % 5 == 0 or frame_idx == self.total_frames:
                    elapsed = max(0.001, time.time() - start_time)
                    current_fps = frame_idx / elapsed
                    self.fps = round(current_fps, 1)
                    if self.total_frames > 0:
                        self.progress = min(99.0, round((frame_idx / self.total_frames) * 100.0, 1))
                        remaining_frames = max(0, self.total_frames - frame_idx)
                        self.eta_seconds = int(remaining_frames / current_fps) if current_fps > 0 else 0

            cap.release()
            try:
                process.stdin.close()
            except Exception:
                pass

            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=5)
                except Exception:
                    pass

            self.status = "completing"
            self.progress = 99.0

            # Re-mux audio if original had audio
            if has_audio and os.path.exists(temp_video_out):
                mux_cmd = [
                    "ffmpeg", "-y",
                    "-i", temp_video_out,
                    "-i", self.input_path,
                    "-map", "0:v:0",
                    "-map", "1:a:0?",
                    "-c:v", "copy",
                    "-c:a", "copy",
                    "-shortest",
                    self.output_path
                ]
                try:
                    mux_res = subprocess.run(mux_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
                    if mux_res.returncode == 0 and os.path.exists(self.output_path) and os.path.getsize(self.output_path) > 0:
                        try:
                            os.remove(temp_video_out)
                        except Exception:
                            pass
                    else:
                        if os.path.exists(self.output_path):
                            os.remove(self.output_path)
                        shutil.move(temp_video_out, self.output_path)
                except Exception:
                    if os.path.exists(self.output_path):
                        os.remove(self.output_path)
                    if os.path.exists(temp_video_out):
                        shutil.move(temp_video_out, self.output_path)
            else:
                if os.path.exists(self.output_path):
                    os.remove(self.output_path)
                if os.path.exists(temp_video_out):
                    shutil.move(temp_video_out, self.output_path)

            self.status = "completed"
            self.progress = 100.0
            self.eta_seconds = 0

        except Exception as e:
            self.status = "failed"
            self.error_message = str(e)
            print(f"Error processing video: {e}")

    def _run_ffmpeg_delogo(self, width: int, height: int, has_audio: bool):
        """
        Runs fast direct FFmpeg delogo filter without frame decoding pipes.
        """
        x = max(0, min(width - 4, self.bbox["x"]))
        y = max(0, min(height - 4, self.bbox["y"]))
        w = max(4, min(width - x, self.bbox["w"]))
        h = max(4, min(height - y, self.bbox["h"]))

        delogo_filter = f"delogo=x={x}:y={y}:w={w}:h={h}:show=0"
        cmd = [
            "ffmpeg", "-y",
            "-i", self.input_path,
            "-vf", delogo_filter,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-pix_fmt", "yuv420p"
        ]
        if has_audio:
            cmd.extend(["-c:a", "copy"])
        else:
            cmd.extend(["-an"])
        cmd.append(self.output_path)

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        while proc.poll() is None:
            if self.is_cancelled:
                proc.terminate()
                if os.path.exists(self.output_path):
                    os.remove(self.output_path)
                return
            line = proc.stdout.readline()
            if "frame=" in line:
                try:
                    parts = line.split("frame=")[1].split()[0]
                    curr = int(parts)
                    self.current_frame = curr
                    if self.total_frames > 0:
                        self.progress = min(99.0, round((curr / self.total_frames) * 100.0, 1))
                except Exception:
                    pass
            time.sleep(0.05)

        proc.wait()
        if proc.returncode == 0:
            self.status = "completed"
            self.progress = 100.0
        else:
            self.status = "failed"
            self.error_message = "FFmpeg delogo processing failed."
