import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File limits & uploads
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 15 * 1024 * 1024))
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# Inference settings
INFER_MAX_DIM = int(os.getenv("INFER_MAX_DIM", 1280))

# Model paths
# Default SVR model path: check backend/models or workspace root
DEFAULT_SVR_PATH = os.path.join(BASE_DIR, "models", "model_berat_udang.pkl")
if not os.path.exists(DEFAULT_SVR_PATH):
    # Fallback to workspace root
    DEFAULT_SVR_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "model_berat_udang.pkl"))

SVR_MODEL_PATH = os.getenv("SVR_MODEL_PATH", DEFAULT_SVR_PATH)

# YOLO model path: weights/yolov8n-seg.pt or weights/yolo26n-seg.pt
DEFAULT_YOLO_PATH = os.path.join(BASE_DIR, "weights", "yolo26n-seg.pt")
if not os.path.exists(DEFAULT_YOLO_PATH):
    # Fallback to yolov8n-seg.pt if custom yolo26n-seg.pt doesn't exist locally
    DEFAULT_YOLO_PATH = os.path.join(BASE_DIR, "weights", "yolov8n-seg.pt")

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", DEFAULT_YOLO_PATH)

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
LOG_FILE_PATH = os.path.join(BASE_DIR, "logs", "shrimp_api.log")

# CORS origins
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
