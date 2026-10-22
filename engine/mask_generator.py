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

def make_gemini_star_template(size: int) -> np.ndarray:
    """
    Generates a synthetic binary template of the 4-pointed Gemini star.
    """
    tmpl = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size / 2.0, size / 2.0
    r = size / 2.0 - 1.0
    theta = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    xs = cx + r * np.sign(np.cos(theta)) * (np.abs(np.cos(theta)) ** 2.0)
    ys = cy + r * np.sign(np.sin(theta)) * (np.abs(np.sin(theta)) ** 2.0)
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    cv2.fillPoly(tmpl, [pts], 255)
    return tmpl

def detect_watermark_from_frame(frame: np.ndarray) -> dict:
    """
    Analyzes the video frame to automatically detect the watermark's location
    (Gemini 4-pointed star, Veo, or other corner watermarks) anywhere in the frame.
    Returns bounding box dict with full logo coverage.
    """
    if frame is None or frame.size == 0:
        return None
        
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # AI video watermarks are placed in corners. We search all 4 corners:
    corners = [
        ('BR', int(w * 0.75), int(h * 0.85), int(w * 0.99), int(h * 0.99)),
        ('TR', int(w * 0.75), int(h * 0.01), int(w * 0.99), int(h * 0.15)),
        ('BL', int(w * 0.01), int(h * 0.85), int(w * 0.25), int(h * 0.99)),
        ('TL', int(w * 0.01), int(h * 0.01), int(w * 0.25), int(h * 0.15)),
    ]
    
    best_candidate = None
    best_star_score = -1.0
    
    # Pass 1: Multi-scale 4-pointed Gemini star template matching
    for cname, x1, y1, x2, y2 in corners:
        crop = gray[y1:y2, x1:x2]
        if crop.shape[0] < 24 or crop.shape[1] < 24:
            continue
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(crop, cv2.MORPH_TOPHAT, k)
        
        weight = 1.15 if cname == 'BR' else 1.0
        
        for tsize in [26, 30, 34, 38]:
            if tophat.shape[0] < tsize or tophat.shape[1] < tsize:
                continue
            tmpl = make_gemini_star_template(tsize)
            res = cv2.matchTemplate(tophat, tmpl, cv2.TM_CCOEFF_NORMED)
            _, maxv, _, maxl = cv2.minMaxLoc(res)
            score = maxv * weight
            if score > best_star_score and maxv > 0.40:
                best_star_score = score
                cx = x1 + maxl[0] + tsize // 2
                cy = y1 + maxl[1] + tsize // 2
                best_candidate = (cx, cy, 'star')
                
    if best_candidate and best_star_score >= 0.45:
        cx, cy, shape = best_candidate
    else:
        # Pass 2: Salient corner feature / Top-Hat peak (e.g. text 'Veo' or non-star logo)
        br_x1, br_y1, br_x2, br_y2 = corners[0][1:]
        crop = gray[br_y1:br_y2, br_x1:br_x2]
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(crop, cv2.MORPH_TOPHAT, k)
        minv, maxv, minl, maxl = cv2.minMaxLoc(tophat)
        if maxv > 25:
            cx = br_x1 + maxl[0]
            cy = br_y1 + maxl[1]
            shape = 'box' if maxv > 150 else 'star'
        else:
            # Fallback to standard bottom-right Gemini placement
            cx = int(w - min(w, h) * 0.070)
            cy = int(h - min(w, h) * 0.040)
            shape = 'star'
            
    # Proportional bounding box guaranteeing 100% full logo coverage
    box_size = int(max(64, min(140, min(w, h) * 0.095)))
    x = max(0, min(w - box_size, int(cx - box_size / 2)))
    y = max(0, min(h - box_size, int(cy - box_size / 2)))
    
    return {
        "x": x,
        "y": y,
        "w": box_size,
        "h": box_size,
        "shape": shape,
        "dilation": 8,
        "feather": 0
    }

def get_default_gemini_bbox(width: int, height: int, frame: np.ndarray = None) -> dict:
    """
    Analyzes the frame to accurately locate the watermark with built-in detection.
    Falls back to bottom-right corner if frame is unavailable.
    """
    if frame is not None:
        detected = detect_watermark_from_frame(frame)
        if detected:
            return detected

    # Safe fallback box
    box_size = int(max(64, min(140, min(width, height) * 0.095)))
    x = max(0, int(width - min(width, height) * 0.070 - box_size / 2))
    y = max(0, int(height - min(width, height) * 0.040 - box_size / 2))
    
    return {
        "x": max(0, min(width - box_size, x)),
        "y": max(0, min(height - box_size, y)),
        "w": box_size,
        "h": box_size,
        "dilation": 8,
        "feather": 0,
        "shape": "star"
    }
