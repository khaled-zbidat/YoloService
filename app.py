# Updated version of your app.py integrating send_to_yolo_service
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image
import sqlite3
import os
import uuid
import boto3
import json
import requests
from typing import Optional

# Disable GPU usage
import torch
torch.cuda.is_available = lambda: False

app = FastAPI()

UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"
DB_PATH = "predictions.db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

# S3 Configuration
AWS_REGION = "eu-central-1"
S3_BUCKET_NAME = "khaledphotosbuckettelegram"
s3 = boto3.client("s3", region_name=AWS_REGION)

# Database init
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_sessions (
                uid TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                original_image TEXT,
                predicted_image TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detection_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_uid TEXT,
                label TEXT,
                score REAL,
                box TEXT,
                FOREIGN KEY (prediction_uid) REFERENCES prediction_sessions (uid)
            )
        """)

init_db()

def download_image_from_s3(image_name: str, local_path: str) -> bool:
    try:
        s3.download_file(S3_BUCKET_NAME, image_name, local_path)
        return True
    except Exception as e:
        print(f"S3 download error: {e}")
        return False

def save_prediction_session(uid, original_image, predicted_image):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO prediction_sessions (uid, original_image, predicted_image)
            VALUES (?, ?, ?)
        """, (uid, original_image, predicted_image))

def save_detection_object(prediction_uid, label, score, box):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO detection_objects (prediction_uid, label, score, box)
            VALUES (?, ?, ?, ?)
        """, (prediction_uid, label, score, str(box)))

def send_to_yolo_service(image_name):
    try:
        yolo_url = os.getenv("YOLO_URL")
        payload = {"image_name": image_name}
        response = requests.post(yolo_url, json=payload, timeout=30)

        if response.status_code == 200:
            return json.loads(response.text)
        else:
            print(f"YOLO service error: {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"YOLO service exception: {e}")
        return None

@app.post("/predict")
async def predict(request: Request):
    uid = str(uuid.uuid4())
    content_type = request.headers.get('content-type', '').lower()

    try:
        if 'application/json' in content_type:
            json_data = await request.json()
            image_name = json_data.get("image_name")
            if not image_name:
                raise HTTPException(status_code=400, detail="Missing 'image_name'")

            ext = os.path.splitext(image_name)[1] or ".jpg"
            original_filename = f"{uid}_original{ext}"
            original_path = os.path.join(UPLOAD_DIR, original_filename)

            if not download_image_from_s3(image_name, original_path):
                raise HTTPException(status_code=404, detail="Image not found in S3")

            prediction_result = send_to_yolo_service(image_name)
            if not prediction_result:
                raise HTTPException(status_code=500, detail="YOLO prediction failed")

            predicted_filename = f"{uid}_predicted{ext}"
            predicted_path = os.path.join(PREDICTED_DIR, predicted_filename)
            predicted_image_data = Image.open(requests.get(prediction_result["predicted_image_url"], stream=True).raw)
            predicted_image_data.save(predicted_path)

            original_s3_key = f"predictions/original_{original_filename}"
            predicted_s3_key = f"predictions/predicted_{predicted_filename}"
            s3.upload_file(original_path, S3_BUCKET_NAME, original_s3_key)
            s3.upload_file(predicted_path, S3_BUCKET_NAME, predicted_s3_key)

            save_prediction_session(uid, original_s3_key, predicted_s3_key)

            for obj in prediction_result.get("detections", []):
                save_detection_object(uid, obj["label"], obj["confidence"], obj["bbox"])

            return JSONResponse({
                "success": True,
                "prediction_uid": uid,
                "source": f"s3:{image_name}",
                "detection_count": len(prediction_result.get("detections", [])),
                "unique_labels": list({d['label'] for d in prediction_result.get("detections", [])}),
                "all_detections": prediction_result.get("detections", []),
                "original_image_s3_key": original_s3_key,
                "predicted_image_s3_key": predicted_s3_key,
                "message": f"Successfully detected {len(prediction_result.get('detections', []))} objects"
            })

        raise HTTPException(status_code=400, detail="Only JSON with image_name supported")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model_used": "external YOLO service"}

@app.get("/predictions/{uid}")
async def get_prediction(uid: str):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            session = conn.execute("""
                SELECT uid, timestamp, original_image, predicted_image
                FROM prediction_sessions WHERE uid = ?
            """, (uid,)).fetchone()

            if not session:
                raise HTTPException(status_code=404, detail="Prediction not found")

            detections = conn.execute("""
                SELECT label, score, box FROM detection_objects 
                WHERE prediction_uid = ? ORDER BY score DESC
            """, (uid,)).fetchall()

            return {
                "uid": session[0],
                "timestamp": session[1],
                "original_image": session[2],
                "predicted_image": session[3],
                "detections": [
                    {"label": d[0], "score": d[1], "box": d[2]} for d in detections
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8667)
