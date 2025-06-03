from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, Response, JSONResponse
from ultralytics import YOLO
from PIL import Image
import sqlite3
import os
import uuid
import shutil
import boto3
import json
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

# Download the AI model (tiny model ~6MB)
model = YOLO("yolov8n.pt")  

# Initialize SQLite (same as before)
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
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_uid ON detection_objects (prediction_uid)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_label ON detection_objects (label)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_score ON detection_objects (score)")

init_db()

# S3 Configuration (same as before)
AWS_REGION = "eu-central-1"
S3_BUCKET_NAME = "khaledphotosbuckettelegram"
s3 = boto3.client("s3")

def download_image_from_s3(image_name: str, local_path: str) -> bool:
    try:
        s3.download_file(S3_BUCKET_NAME, image_name, local_path)
        return True
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        return False

def upload_image_to_s3(local_path: str, image_name: str) -> bool:
    try:
        s3.upload_file(local_path, S3_BUCKET_NAME, image_name)
        return True
    except Exception as e:
        print(f"Error uploading to S3: {e}")
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

@app.post("/predict")
async def predict(request: Request):
    """
    New version that properly handles both:
    - JSON requests with image_name (from your bot)
    - File uploads (for other clients)
    """
    # Generate unique ID for this prediction
    uid = str(uuid.uuid4())
    
    # Check content type to determine how to process
    content_type = request.headers.get('content-type', '')
    
    if 'application/json' in content_type:
        # Handle JSON request (from your bot)
        try:
            json_data = await request.json()
            image_name = json_data.get("image_name")
            
            if not image_name:
                raise HTTPException(status_code=400, detail="image_name field is required in JSON body")
            
            print(f"Processing image from S3: {image_name}")
            ext = os.path.splitext(image_name)[1] or ".jpg"
            original_filename = uid + ext
            original_path = os.path.join(UPLOAD_DIR, original_filename)
            
            # Download from S3
            if not download_image_from_s3(image_name, original_path):
                raise HTTPException(status_code=404, detail="Image not found in S3")
            
            return await process_prediction(uid, original_path, ext)
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON request: {str(e)}")
    
    else:
        # Handle file upload (for backward compatibility)
        try:
            form_data = await request.form()
            if 'file' not in form_data:
                raise HTTPException(status_code=400, detail="Either send JSON with image_name or upload a file")
            
            file = form_data['file']
            if not file.filename:
                raise HTTPException(status_code=400, detail="No file uploaded")
            
            print(f"Processing uploaded file: {file.filename}")
            ext = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
            original_filename = uid + ext
            original_path = os.path.join(UPLOAD_DIR, original_filename)
            
            # Save uploaded file
            with open(original_path, "wb") as f:
                contents = await file.read()
                f.write(contents)
            
            return await process_prediction(uid, original_path, ext)
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing file upload: {str(e)}")

async def process_prediction(uid, original_path, ext):
    """
    Common prediction processing for both JSON and file upload paths
    """
    try:
        predicted_filename = uid + ext
        predicted_path = os.path.join(PREDICTED_DIR, predicted_filename)

        print(f"Running YOLO prediction on: {original_path}")
        results = model(original_path, device="cpu")

        # Create annotated image
        annotated_frame = results[0].plot()
        annotated_image = Image.fromarray(annotated_frame)
        annotated_image.save(predicted_path)

        # Upload to S3
        original_s3_key = f"predictions/original_{os.path.basename(original_path)}"
        predicted_s3_key = f"predictions/predicted_{predicted_filename}"
        upload_image_to_s3(original_path, original_s3_key)
        upload_image_to_s3(predicted_path, predicted_s3_key)

        # Save to database
        save_prediction_session(uid, original_s3_key, predicted_s3_key)

        # Process detection results
        detected_labels = []
        for box in results[0].boxes:
            label_idx = int(box.cls[0].item())
            label = model.names[label_idx]
            score = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            save_detection_object(uid, label, score, bbox)
            detected_labels.append(label)

        return JSONResponse({
            "prediction_uid": uid,
            "detection_count": len(results[0].boxes),
            "labels": detected_labels,
            "original_image_s3_key": original_s3_key,
            "predicted_image_s3_key": predicted_s3_key,
            "message": f"Successfully detected {len(detected_labels)} objects: {', '.join(set(detected_labels))}"
        })
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# [Rest of your endpoints remain the same...]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8667)