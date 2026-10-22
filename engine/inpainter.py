import os
import cv2
import numpy as np
import requests
import sys

LAMA_MODEL_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
LAMA_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "lama_fp32.onnx")

class Inpainter:
    def inpaint(self, frame: np.ndarray, mask: np.ndarray, bbox: dict = None) -> np.ndarray:
        raise NotImplementedError

class TeleaInpainter(Inpainter):
    def __init__(self, radius: int = 3):
        self.radius = radius

    def inpaint(self, frame: np.ndarray, mask: np.ndarray, bbox: dict = None) -> np.ndarray:
        if bbox:
            # Crop optimization
            pad = 12
            h, w = frame.shape[:2]
            x1 = max(0, bbox['x'] - pad)
            y1 = max(0, bbox['y'] - pad)
            x2 = min(w, bbox['x'] + bbox['w'] + pad)
            y2 = min(h, bbox['y'] + bbox['h'] + pad)
            
            crop_frame = frame[y1:y2, x1:x2]
            crop_mask = mask[y1:y2, x1:x2]
            if np.any(crop_mask > 0):
                inpainted_crop = cv2.inpaint(crop_frame, crop_mask, self.radius, cv2.INPAINT_TELEA)
                result = frame.copy()
                result[y1:y2, x1:x2] = inpainted_crop
                return result
            return frame
        return cv2.inpaint(frame, mask, self.radius, cv2.INPAINT_TELEA)

class NavierStokesInpainter(Inpainter):
    def __init__(self, radius: int = 3):
        self.radius = radius

    def inpaint(self, frame: np.ndarray, mask: np.ndarray, bbox: dict = None) -> np.ndarray:
        if bbox:
            pad = 12
            h, w = frame.shape[:2]
            x1 = max(0, bbox['x'] - pad)
            y1 = max(0, bbox['y'] - pad)
            x2 = min(w, bbox['x'] + bbox['w'] + pad)
            y2 = min(h, bbox['y'] + bbox['h'] + pad)
            
            crop_frame = frame[y1:y2, x1:x2]
            crop_mask = mask[y1:y2, x1:x2]
            if np.any(crop_mask > 0):
                inpainted_crop = cv2.inpaint(crop_frame, crop_mask, self.radius, cv2.INPAINT_NS)
                result = frame.copy()
                result[y1:y2, x1:x2] = inpainted_crop
                return result
            return frame
        return cv2.inpaint(frame, mask, self.radius, cv2.INPAINT_NS)

class BlurInpainter(Inpainter):
    def __init__(self, blur_ksize: int = 25):
        self.blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1

    def inpaint(self, frame: np.ndarray, mask: np.ndarray, bbox: dict = None) -> np.ndarray:
        # Gaussian blur filtered with soft mask blending
        blurred = cv2.GaussianBlur(frame, (self.blur_ksize, self.blur_ksize), 0)
        # Soften mask edges for seamless blending
        soft_mask = cv2.GaussianBlur(mask, (15, 15), 0).astype(np.float32) / 255.0
        soft_mask = np.expand_dims(soft_mask, axis=2)
        
        result = (frame.astype(np.float32) * (1.0 - soft_mask) + blurred.astype(np.float32) * soft_mask)
        return np.clip(result, 0, 255).astype(np.uint8)

def _setup_cuda_dll_paths():
    """
    Ensures NVIDIA CUDA/cuDNN DLL directories from site-packages are added to Windows DLL search path.
    """
    try:
        import onnxruntime as ort
        site_pkg = os.path.dirname(os.path.dirname(os.path.abspath(ort.__file__)))
        nvidia_dir = os.path.join(site_pkg, 'nvidia')
        if os.path.isdir(nvidia_dir):
            for sub in os.listdir(nvidia_dir):
                bin_dir = os.path.join(nvidia_dir, sub, 'bin')
                if os.path.isdir(bin_dir):
                    try:
                        os.add_dll_directory(bin_dir)
                    except Exception:
                        pass
                    if bin_dir not in os.environ.get('PATH', ''):
                        os.environ['PATH'] = bin_dir + os.pathsep + os.environ['PATH']
    except Exception:
        pass

_setup_cuda_dll_paths()

_SHARED_LAMA_SESSION = None
_SHARED_LAMA_INPUT_NAMES = None
_SHARED_LAMA_OUTPUT_NAME = None

class LamaInpainter(Inpainter):
    def __init__(self, model_path: str = LAMA_MODEL_PATH):
        self.model_path = model_path
        self.session = None
        self.input_names = []
        self.output_name = ""
        self._init_session()

    def _init_session(self):
        global _SHARED_LAMA_SESSION, _SHARED_LAMA_INPUT_NAMES, _SHARED_LAMA_OUTPUT_NAME
        if _SHARED_LAMA_SESSION is not None:
            self.session = _SHARED_LAMA_SESSION
            self.input_names = _SHARED_LAMA_INPUT_NAMES
            self.output_name = _SHARED_LAMA_OUTPUT_NAME
            return

        if not os.path.exists(self.model_path):
            return
        
        _setup_cuda_dll_paths()
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        # Prioritize NVIDIA CUDA GPU acceleration, with CPU fallback
        providers = []
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        cpu_count = os.cpu_count() or 4
        sess_options.intra_op_num_threads = min(8, cpu_count)

        self.session = ort.InferenceSession(self.model_path, sess_options=sess_options, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name

        _SHARED_LAMA_SESSION = self.session
        _SHARED_LAMA_INPUT_NAMES = self.input_names
        _SHARED_LAMA_OUTPUT_NAME = self.output_name

    def is_available(self) -> bool:
        return self.session is not None

    def inpaint(self, frame: np.ndarray, mask: np.ndarray, bbox: dict = None) -> np.ndarray:
        if not self.is_available():
            # Fallback to Telea if model not yet downloaded
            return TeleaInpainter().inpaint(frame, mask, bbox)

        h, w = frame.shape[:2]
        
        # Crop context area around watermark for high performance and best resolution preservation
        if bbox:
            pad = int(max(bbox['w'], bbox['h']) * 1.5)
            x1 = max(0, bbox['x'] - pad)
            y1 = max(0, bbox['y'] - pad)
            x2 = min(w, bbox['x'] + bbox['w'] + pad)
            y2 = min(h, bbox['y'] + bbox['h'] + pad)
        else:
            # Fallback to bottom right quadrant
            x1, y1 = int(w * 0.7), int(h * 0.7)
            x2, y2 = w, h

        crop_img = frame[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]

        ch, cw = crop_img.shape[:2]
        if ch < 4 or cw < 4 or not np.any(crop_mask > 0):
            return frame

        # Resize crop and mask to 512x512 as strictly required by this ONNX model
        interp = cv2.INTER_AREA if (cw > 512 or ch > 512) else cv2.INTER_CUBIC
        model_input_img = cv2.resize(crop_img, (512, 512), interpolation=interp)
        model_input_mask = cv2.resize(crop_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        # Prepare inputs for LaMa:
        # Image: RGB, float32, [0, 1], shape (1, 3, 512, 512)
        rgb_img = cv2.cvtColor(model_input_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = np.transpose(rgb_img, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        # Mask: float32, binary (1.0 for mask, 0.0 for keep), shape (1, 1, 512, 512)
        mask_binary = (model_input_mask > 127).astype(np.float32)
        mask_tensor = mask_binary[np.newaxis, np.newaxis, ...]

        inputs = {
            self.input_names[0]: img_tensor,
            self.input_names[1]: mask_tensor
        }

        try:
            outputs = self.session.run([self.output_name], inputs)
            res_tensor = outputs[0][0]  # shape (3, 512, 512)
            
            # Clip and convert back to uint8 BGR
            res_tensor = np.clip(res_tensor, 0, 255).astype(np.uint8)
            res_rgb = np.transpose(res_tensor, (1, 2, 0))
            res_bgr = cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)

            # Resize inpainted 512x512 patch back to original crop size (cw, ch)
            res_bgr = cv2.resize(res_bgr, (cw, ch), interpolation=cv2.INTER_CUBIC)

            # Blend with soft transition around the mask to avoid any seam
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            feather_mask = cv2.dilate(crop_mask, kernel, iterations=1)
            feather_mask = cv2.GaussianBlur(feather_mask, (5, 5), 0).astype(np.float32) / 255.0
            feather_mask = np.expand_dims(feather_mask, axis=2)

            blended_crop = (frame[y1:y2, x1:x2].astype(np.float32) * (1.0 - feather_mask) + 
                            res_bgr.astype(np.float32) * feather_mask).astype(np.uint8)

            result = frame.copy()
            result[y1:y2, x1:x2] = blended_crop
            return result
        except Exception as e:
            print(f"Warning: LaMa model inference error: {e}, falling back to Telea.")
            return TeleaInpainter().inpaint(frame, mask, bbox)

def download_lama_model(progress_callback=None) -> bool:
    """
    Downloads the LaMa ONNX model with progress tracking.
    """
    os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)
    temp_path = LAMA_MODEL_PATH + ".download"
    
    if os.path.exists(LAMA_MODEL_PATH) and os.path.getsize(LAMA_MODEL_PATH) > 100_000_000:
        return True

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(LAMA_MODEL_URL, stream=True, headers=headers, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks

        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total_size > 0:
                        progress_callback(downloaded, total_size, (downloaded / total_size) * 100.0)

        if os.path.exists(LAMA_MODEL_PATH):
            os.remove(LAMA_MODEL_PATH)
        os.rename(temp_path, LAMA_MODEL_PATH)
        return True
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e
