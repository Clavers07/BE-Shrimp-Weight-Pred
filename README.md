# 🦐 Backend Flask API — Shrimp Weight Prediction

Backend API Flask untuk segmentasi udang (YOLOv8-Seg / YOLO26n-Seg) dan estimasi berat udang (Support Vector Regression `.pkl`).

## 🚀 Cara Menjalankan Backend

### 1. Buat & Aktivasi Virtual Environment
```powershell
# Masuk ke folder backend
cd backend

# Buat virtual environment (hanya dilakukan sekali saat pertama kali clone/fork)
python -m venv venv

# Aktivasi Virtual Environment
.\venv\Scripts\activate       # Windows (PowerShell)
# source venv/bin/activate   # Linux / macOS
```

### 2. Install Dependensi
```powershell
pip install -r requirements.txt
```

### 3. Setup Environment Variables (`.env`)
```powershell
# Copy template .env.example menjadi .env
copy .env.example .env        # Windows
# cp .env.example .env        # Linux / macOS
```

### 4. Jalankan Server Flask
```powershell
python app.py
```
Server akan berjalan di `http://127.0.0.1:5000`.

---

## 🧪 Cara Pengujian / Smoke Test

Jalankan skrip pengujian otomatis:
```powershell
python test_api.py
```

Skrip ini akan menguji:
1. `GET /healthz` - Status kesiapan model & kombinasi fitur aktif.
2. `POST /api/v1/shrimp/predict-weight` - Kasus validasi error (missing key, format tidak didukung).
3. `POST /api/v1/shrimp/predict-weight` - Inferensi gambar dan kecocokan skema JSON kontrak `API_CONTRACT.md`.

---

## 📡 Endpoints API

### 1. Healthcheck (`GET /healthz`)
Mengembalikan versi model dan kombinasi fitur SVR yang sedang aktif.

**Response 200 OK:**
```json
{
  "status": "ok",
  "model_yolo_version": "yolov8n-seg",
  "model_svr_combination": "area-perimeter-length",
  "feature_order": ["area", "perimeter", "length"]
}
```

### 2. Predict Shrimp Weight (`POST /api/v1/shrimp/predict-weight`)
- **Content-Type**: `multipart/form-data`
- **Body Key**: `image` (File `.jpg`, `.jpeg`, `.png`, max 15MB)
- **Header**: `X-Request-Id` (Opsional, UUID v4)

**Response 200 OK:**
```json
{
  "status": "success",
  "request_id": "b3f1c2a0-...",
  "meta": {
    "processing_time_ms": 320,
    "model_yolo_version": "yolov8n-seg",
    "model_svr_combination": "area-perimeter-length"
  },
  "data": {
    "total_detected": 1,
    "predictions": [
      {
        "id": 1,
        "berat_gram": 11.52,
        "features_extracted": {
          "area": 25450.5,
          "perimeter": 890.4,
          "length": 320.2
        },
        "polygon_coordinates": [[130, 350], [134, 352]]
      }
    ]
  }
}
```

---

## 📜 Log Output

Log backend disimpan secara otomatis di file [`backend/logs/shrimp_api.log`](file:///c:/Muhammad%20Kalam%20Sabili-TRPL%20B-2023/Semester%207/Activity/Shrimp%20Weight%20Prediction/backend/logs/shrimp_api.log) dengan format terstruktur menyertakan `request_id` pada tiap baris.
