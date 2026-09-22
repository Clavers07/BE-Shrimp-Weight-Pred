import cv2
import numpy as np

def extract_area(pts: np.ndarray) -> float:
    return float(cv2.contourArea(pts))

def extract_perimeter(pts: np.ndarray) -> float:
    return float(cv2.arcLength(pts, True))

def extract_bbox_dims(pts: np.ndarray):
    """
    Extracts length and width using minAreaRect (rotated bounding box).
    Returns (length, width) where length >= width.
    """
    rect = cv2.minAreaRect(pts)
    (_, _), (w, h), _ = rect
    length = max(w, h)
    width = min(w, h)
    return float(length), float(width)

def extract_solidity(pts: np.ndarray) -> float:
    area = cv2.contourArea(pts)
    hull = cv2.convexHull(pts)
    hull_area = cv2.contourArea(hull)
    return float(area / hull_area) if hull_area > 0 else 0.0

def extract_aspect_ratio(pts: np.ndarray) -> float:
    length, width = extract_bbox_dims(pts)
    return float(length / width) if width > 0 else 0.0

def extract_all_features(pts: np.ndarray) -> dict:
    """
    pts must be float32, shape (N, 2).
    Extracts all 6 geometrical features into a dictionary.
    """
    length, width = extract_bbox_dims(pts)
    area = extract_area(pts)
    perimeter = extract_perimeter(pts)
    solidity = extract_solidity(pts)
    aspect_ratio = extract_aspect_ratio(pts)

    return {
        "area": area,
        "perimeter": perimeter,
        "length": length,
        "width": width,
        "solidity": solidity,
        "aspect_ratio": aspect_ratio,
    }
