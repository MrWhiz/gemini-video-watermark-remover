import numpy as np
import cv2

def create_gemini_star_contour(cx: float, cy: float, radius: float, num_points: int = 120) -> np.ndarray:
    """
    Generates the 4-pointed astroid / star contour characteristic of the Gemini AI logo.
    Equation: x = R * cos^3(t), y = R * sin^3(t)
    """
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    # Astroid formula produces 4 sharp points at 0, 90, 180, 270 degrees
    # with smooth concave curves connecting them.
    # We slightly tune the exponent to match Gemini's subtle curve profile (exponent ~ 3.0)
    # Curve exponent: 2.0 provides a fuller star contour matching the Gemini logo's true shape
    # and preventing thin arm cutoffs
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    xs = cx + radius * np.sign(cos_t) * (np.abs(cos_t) ** 2.0)
    ys = cy + radius * np.sign(sin_t) * (np.abs(sin_t) ** 2.0)
    
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    return pts

def generate_mask(
    height: int,
    width: int,
    x: int,
    y: int,
    w: int,
    h: int,
    shape: str = "star",
    dilation: int = 6,
    feather: int = 0
) -> np.ndarray:
    """
    Generates a single-channel uint8 binary mask (0 = keep, 255 = inpaint).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Clamp bounding box
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    w = max(4, min(width - x, w))
    h = max(4, min(height - y, h))
    
    if shape == "star":
        cx = x + w / 2.0
        cy = y + h / 2.0
        radius = min(w, h) / 2.0
        pts = create_gemini_star_contour(cx, cy, radius)
        cv2.fillPoly(mask, [pts], 255)
    elif shape == "circle":
        cx = int(x + w / 2.0)
        cy = int(y + h / 2.0)
        radius = int(min(w, h) / 2.0)
        cv2.circle(mask, (cx, cy), radius, 255, -1)
    else:  # rectangle / box
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        
    # Apply dilation to guarantee full coverage of outer tips and anti-aliased edge glow
    if dilation > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
        
    if feather > 0:
        ksize = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        
    return mask

def get_default_gemini_bbox(width: int, height: int) -> dict:
    """
    Returns an initial default bounding box for the mask.
    The user can click and drag anywhere on the video frame to place it.
    """
    box_size = int(max(64, min(140, min(width, height) * 0.095)))
    x = max(0, width - box_size - 40)
    y = max(0, height - box_size - 40)
    
    return {
        "x": x,
        "y": y,
        "w": box_size,
        "h": box_size,
        "dilation": 8,
        "feather": 0,
        "shape": "star"
    }
