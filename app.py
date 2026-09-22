import os
import time
import uuid
import cv2
import joblib
import numpy as np
from flask import Flask, request, jsonify, g
from flask_cors import CORS

import config
from logger import logger as log
from feature_extractor import extract_all_features

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
CORS(app, origins=config.CORS_ORIGINS)

# Global model state
model_yolo = None
model_svr = None
FEATURE_ORDER = None
SVR_COMBINATION_NAME = None
YOLO_VERSION_NAME = "yolo26n-seg"


def load_models():
    global model_yolo, model_svr, FEATURE_ORDER, SVR_COMBINATION_NAME, YOLO_VERSION_NAME
    t0 = time.time()
    
    # 1. Load YOLO model
    yolo_path = config.YOLO_MODEL_PATH
    log.info(f"[BOOT] Memuat model YOLO dari: {yolo_path}")
    from ultralytics import YOLO
    try:
        model_yolo = YOLO(yolo_path)
        YOLO_VERSION_NAME = os.path.basename(yolo_path).replace(".pt", "")
    except Exception as e:
        log.warning(f"[BOOT] Gagal memuat {yolo_path} ({e}), mencoba fallback 'yolov8n-seg.pt'...")
        model_yolo = YOLO("yolov8n-seg.pt")
        YOLO_VERSION_NAME = "yolov8n-seg"

    # 2. Load SVR model (.pkl)
    svr_path = config.SVR_MODEL_PATH
    log.info(f"[BOOT] Memuat model SVR (.pkl) dari: {svr_path}")
    if not os.path.exists(svr_path):
        raise RuntimeError(f"File model SVR tidak ditemukan di path: {svr_path}")
    
    model_svr = joblib.load(svr_path)

    # 3. Dynamic Feature Ordering check (§2 BE_INSTRUCTIONS.md)
    if hasattr(model_svr, "feature_names_in_"):
        FEATURE_ORDER = [f.lower() for f in model_svr.feature_names_in_]
        log.info(f"[BOOT] FEATURE_ORDER diambil otomatis dari model: {FEATURE_ORDER}")
    else:
        log.error("[BOOT] Model SVR tidak memiliki atribut feature_names_in_.")
        raise RuntimeError(
            "Model .pkl tidak punya feature_names_in_ (kemungkinan sklearn lama / "
            "di-fit dari numpy array). Urutan fitur harus dikonfirmasi manual ke tim "
            "model sebelum lanjut deploy — JANGAN menebak dari nama folder."
        )
    
    SVR_COMBINATION_NAME = "-".join(FEATURE_ORDER)

    # 4. Warm-up inference
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model_yolo(dummy, verbose=False)

    log.info(f"[BOOT] Models loaded successfully! YOLO={YOLO_VERSION_NAME}, "
             f"FEATURE_ORDER={FEATURE_ORDER}, waktu_load={time.time()-t0:.2f}s")


@app.before_request
def attach_request_id():
    g.req_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))
    g.t_start = time.time()


def err(code, error_code, message):
    req_id = getattr(g, "req_id", "NO_REQ_ID")
    log.warning(f"[{req_id}] {error_code}: {message}")
    return jsonify({
        "status": "error",
        "request_id": req_id,
        "error_code": error_code,
        "message": message
    }), code


@app.route('/healthz', methods=['GET'])
def healthz():
    ready = model_yolo is not None and model_svr is not None
    if not ready:
        return jsonify({
            "status": "not_ready",
            "error_code": "MODEL_NOT_READY",
            "message": "Model belum selesai dimuat"
        }), 503

    return jsonify({
        "status": "ok",
        "model_yolo_version": YOLO_VERSION_NAME,
        "model_svr_combination": SVR_COMBINATION_NAME,
        "feature_order": FEATURE_ORDER,
    }), 200


@app.route('/api/v1/shrimp/predict-weight', methods=['POST'])
def predict_shrimp_weight():
    req_id = getattr(g, "req_id", str(uuid.uuid4()))
    t_start = getattr(g, "t_start", time.time())
    log.info(f"[{req_id}] START predict-weight")

    if model_yolo is None or model_svr is None:
        return err(503, "MODEL_NOT_READY", "Model server belum siap memproses inferensi")

    # 1. Validate File presence
    if 'image' not in request.files:
        return err(400, "NO_FILE_KEY", "Key 'image' tidak ditemukan di request")

    file = request.files['image']
    if file.filename == '':
        return err(400, "EMPTY_FILE", "File gambar tidak boleh kosong")

    # 2. Validate Extension
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in config.ALLOWED_EXTENSIONS:
        return err(415, "UNSUPPORTED_TYPE", f"Ekstensi .{ext} tidak didukung")

    # 3. Read Buffer & Validate Size
    filestr = file.read()
    if len(filestr) > config.MAX_CONTENT_LENGTH:
        return err(413, "FILE_TOO_LARGE", "Ukuran gambar melebihi 15MB")

    t_decode = time.time()
    np_img = np.frombuffer(filestr, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img is None:
        return err(422, "INVALID_IMAGE", "Gambar tidak dapat dibaca / rusak")
    
    log.debug(f"[{req_id}] decode ok shape={img.shape} t={time.time()-t_decode:.3f}s")

    try:
        original_h, original_w = img.shape[:2]

        # 4. Resize for YOLO Inference maintaining aspect ratio (§4 & §6 BE_INSTRUCTIONS.md)
        max_dim = max(original_h, original_w)
        scale = min(1.0, config.INFER_MAX_DIM / float(max_dim))
        if scale < 1.0:
            infer_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            infer_img = img

        infer_h, infer_w = infer_img.shape[:2]
        scale_x = float(original_w) / float(infer_w)
        scale_y = float(original_h) / float(infer_h)

        t_infer = time.time()
        results = model_yolo(infer_img, verbose=False)
        r = results[0]  # Fix bug #1: results[0] for list of Results
        infer_ms = (time.time() - t_infer) * 1000
        
        detected_count = 0 if r.masks is None else len(r.masks.xy)
        log.info(f"[{req_id}] YOLO inference {infer_ms:.1f}ms, detected={detected_count}")

        if r.masks is None or detected_count == 0:
            total_ms = int((time.time() - t_start) * 1000)
            return jsonify({
                "status": "success",
                "request_id": req_id,
                "meta": {
                    "processing_time_ms": total_ms,
                    "model_yolo_version": YOLO_VERSION_NAME,
                    "model_svr_combination": SVR_COMBINATION_NAME,
                },
                "data": {"total_detected": 0, "predictions": []}
            }), 200

        predictions_list = []
        for idx, yolo_poly in enumerate(r.masks.xy):
            pts = yolo_poly.astype(np.float32)

            # Rescale polygon coordinates back to original image resolution (§4 BE_INSTRUCTIONS.md)
            if scale < 1.0:
                pts[:, 0] *= scale_x
                pts[:, 1] *= scale_y

            extracted_map = extract_all_features(pts)
            input_features = [extracted_map[f] for f in FEATURE_ORDER]
            input_matrix = np.array([input_features], dtype=np.float64)

            t_pred = time.time()
            predicted_weight = model_svr.predict(input_matrix)
            pred_val = float(predicted_weight[0])
            pred_ms = (time.time() - t_pred) * 1000

            log.debug(f"[{req_id}] obj#{idx+1} features={input_features} "
                      f"berat={pred_val:.2f}g t={pred_ms:.1f}ms")

            predictions_list.append({
                "id": idx + 1,
                "berat_gram": float(round(pred_val, 2)),
                "features_extracted": {f: float(round(extracted_map[f], 2)) for f in FEATURE_ORDER},
                "polygon_coordinates": pts.astype(int).tolist(),
            })

        total_ms = int((time.time() - t_start) * 1000)
        log.info(f"[{req_id}] DONE total_detected={len(predictions_list)} t_total={total_ms}ms")

        return jsonify({
            "status": "success",
            "request_id": req_id,
            "meta": {
                "processing_time_ms": total_ms,
                "model_yolo_version": YOLO_VERSION_NAME,
                "model_svr_combination": SVR_COMBINATION_NAME,
            },
            "data": {
                "total_detected": len(predictions_list),
                "predictions": predictions_list
            }
        }), 200

    except Exception as e:
        log.exception(f"[{req_id}] INFERENCE_FAILED: {str(e)}")
        return err(500, "INFERENCE_FAILED", "Terjadi kesalahan internal saat memproses gambar")


# Load models at module import / startup
try:
    load_models()
except Exception as e:
    log.error(f"[BOOT] Gagal menginisialisasi model saat startup: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
