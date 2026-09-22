import os
import sys
import time
import requests
import numpy as np
import cv2

BASE_URL = "http://127.0.0.1:5000"

def create_dummy_shrimp_image(filename="test_shrimp.jpg"):
    """
    Creates a dummy image with a drawn white ellipse on black background 
    to simulate an object for segmentation tests.
    """
    img = np.zeros((800, 1000, 3), dtype=np.uint8)
    cv2.ellipse(img, (500, 400), (200, 80), 30, 0, 360, (255, 255, 255), -1)
    cv2.imwrite(filename, img)
    return filename

def run_tests():
    print("=== START BACKEND SMOKE TESTS ===")
    
    # 1. Test Healthz
    print("\n1. Testing GET /healthz ...")
    try:
        res = requests.get(f"{BASE_URL}/healthz")
        print(f"Status Code: {res.status_code}")
        print(f"Response: {res.json()}")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    except Exception as e:
        print(f"Healthz check failed: {e}")
        return False

    # 2. Test Error - Missing File Key
    print("\n2. Testing POST /api/v1/shrimp/predict-weight (Missing 'image' key) ...")
    res = requests.post(f"{BASE_URL}/api/v1/shrimp/predict-weight", files={"wrong_key": ("dummy.jpg", b"123")})
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.json()}")
    assert res.status_code == 400
    assert res.json()["error_code"] == "NO_FILE_KEY"

    # 3. Test Error - Unsupported Extension
    print("\n3. Testing POST /api/v1/shrimp/predict-weight (Unsupported extension) ...")
    res = requests.post(f"{BASE_URL}/api/v1/shrimp/predict-weight", files={"image": ("dummy.txt", b"hello text")})
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.json()}")
    assert res.status_code == 415
    assert res.json()["error_code"] == "UNSUPPORTED_TYPE"

    # 4. Test Valid Image Prediction
    print("\n4. Testing POST /api/v1/shrimp/predict-weight (Valid image upload) ...")
    test_img_path = create_dummy_shrimp_image("test_shrimp_temp.jpg")
    try:
        with open(test_img_path, "rb") as f:
            headers = {"X-Request-Id": "test-req-uuid-12345"}
            res = requests.post(
                f"{BASE_URL}/api/v1/shrimp/predict-weight",
                files={"image": ("test_shrimp.jpg", f, "image/jpeg")},
                headers=headers
            )
        print(f"Status Code: {res.status_code}")
        print(f"Response JSON: {res.json()}")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["request_id"] == "test-req-uuid-12345"
        assert "meta" in data
        assert "data" in data
        assert "total_detected" in data["data"]
        assert "predictions" in data["data"]
        print("[SUCCESS] Response schema contract validated successfully!")
    finally:
        if os.path.exists(test_img_path):
            os.remove(test_img_path)

    print("\n=== ALL TESTS PASSED SUCCESSFULLY! ===")
    return True

if __name__ == "__main__":
    run_tests()
