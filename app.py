from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, Response
from ultralytics import YOLO
from PIL import Image
import sqlite3
import os
import uuid
import shutil
import boto3
import json
from typing import Optional

#S3
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

# Initialize SQLite
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        # Create the predictions main table to store the prediction session
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_sessions (
                uid TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                original_image TEXT,
                predicted_image TEXT
            )
        """)
        
        # Create the objects table to store individual detected objects in a given image
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
        
        # Create index for faster queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_uid ON detection_objects (prediction_uid)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_label ON detection_objects (label)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_score ON detection_objects (score)")

init_db()
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
    """
    Save prediction session to database
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO prediction_sessions (uid, original_image, predicted_image)
            VALUES (?, ?, ?)
        """, (uid, original_image, predicted_image))

def save_detection_object(prediction_uid, label, score, box):
    """
    Save detection object to database
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO detection_objects (prediction_uid, label, score, box)
            VALUES (?, ?, ?, ?)
        """, (prediction_uid, label, score, str(box)))

@app.post("/predict")
async def predict(
    request: Request,
    file: Optional[UploadFile] = File(None)
):
    """
    Predict endpoint with fallback logic:
    1. First check for image_name in request body (JSON)
    2. If not found, check for uploaded file
    3. If neither found, return 400 error
    """
    
    # Generate unique ID for this prediction
    uid = str(uuid.uuid4())
    
    # Try to read JSON body first (for image_name)
    image_name = None
    try:
        body = await request.body()
        if body:
            json_data = json.loads(body.decode())
            image_name = json_data.get("image_name")
    except (json.JSONDecodeError, UnicodeDecodeError):
        # If JSON parsing fails, continue to check for file upload
        pass
    
    # FALLBACK LOGIC:
    if image_name:
        # Case 1: Image name provided - download from S3
        print(f"Processing image from S3: {image_name}")
        ext = os.path.splitext(image_name)[1] or ".jpg"
        original_filename = uid + ext
        original_path = os.path.join(UPLOAD_DIR, original_filename)
        
        success = download_image_from_s3(image_name, original_path)
        if not success:
            raise HTTPException(status_code=404, detail="Image not found in S3")
            
    elif file:
        # Case 2: File uploaded - save locally
        print(f"Processing uploaded file: {file.filename}")
        ext = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
        original_filename = uid + ext
        original_path = os.path.join(UPLOAD_DIR, original_filename)
        
        with open(original_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
    else:
        # Case 3: Neither image_name nor file provided
        raise HTTPException(
            status_code=400, 
            detail="Bad request: Please provide either 'image_name' in JSON body or upload a file"
        )

    # Process the image with YOLO
    try:
        predicted_filename = uid + ext
        predicted_path = os.path.join(PREDICTED_DIR, predicted_filename)

        print(f"Running YOLO prediction on: {original_path}")
        results = model(original_path, device="cpu")

        # Create annotated image
        annotated_frame = results[0].plot()
        annotated_image = Image.fromarray(annotated_frame)
        annotated_image.save(predicted_path)

        # Upload both original and predicted images to S3
        original_s3_key = f"predictions/original_{original_filename}"
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

        return {
            "prediction_uid": uid,
            "detection_count": len(results[0].boxes),
            "labels": detected_labels,
            "original_image_s3_key": original_s3_key,
            "predicted_image_s3_key": predicted_s3_key,
            "message": f"Successfully detected {len(detected_labels)} objects: {', '.join(set(detected_labels))}"
        }
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/prediction/{uid}")
def get_prediction_by_uid(uid: str):
    """
    Get prediction session by uid with all detected objects
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        # Get prediction session
        session = conn.execute("SELECT * FROM prediction_sessions WHERE uid = ?", (uid,)).fetchone()
        if not session:
            raise HTTPException(status_code=404, detail="Prediction not found")
            
        # Get all detection objects for this prediction
        objects = conn.execute(
            "SELECT * FROM detection_objects WHERE prediction_uid = ?", 
            (uid,)
        ).fetchall()
        
        return {
            "uid": session["uid"],
            "timestamp": session["timestamp"],
            "original_image": session["original_image"],
            "predicted_image": session["predicted_image"],
            "detection_objects": [
                {
                    "id": obj["id"],
                    "label": obj["label"],
                    "score": obj["score"],
                    "box": obj["box"]
                } for obj in objects
            ]
        }

@app.get("/predictions/label/{label}")
def get_predictions_by_label(label: str):
    """
    Get prediction sessions containing objects with specified label
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT DISTINCT ps.uid, ps.timestamp
            FROM prediction_sessions ps
            JOIN detection_objects do ON ps.uid = do.prediction_uid
            WHERE do.label = ?
        """, (label,)).fetchall()
        
        return [{"uid": row["uid"], "timestamp": row["timestamp"]} for row in rows]

@app.get("/predictions/score/{min_score}")
def get_predictions_by_score(min_score: float):
    """
    Get prediction sessions containing objects with score >= min_score
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT DISTINCT ps.uid, ps.timestamp
            FROM prediction_sessions ps
            JOIN detection_objects do ON ps.uid = do.prediction_uid
            WHERE do.score >= ?
        """, (min_score,)).fetchall()
        
        return [{"uid": row["uid"], "timestamp": row["timestamp"]} for row in rows]

@app.get("/image/{type}/{filename}")
def get_image(type: str, filename: str):
    """
    Get image by type and filename
    """
    if type not in ["original", "predicted"]:
        raise HTTPException(status_code=400, detail="Invalid image type")
    path = os.path.join("uploads", type, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)

@app.get("/prediction/{uid}/image")
def get_prediction_image(uid: str, request: Request):
    """
    Get prediction image by uid
    """
    accept = request.headers.get("accept", "")
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT predicted_image FROM prediction_sessions WHERE uid = ?", (uid,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Prediction not found")
        image_path = row[0]

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Predicted image file not found")

    if "image/png" in accept:
        return FileResponse(image_path, media_type="image/png")
    elif "image/jpeg" in accept or "image/jpg" in accept:
        return FileResponse(image_path, media_type="image/jpeg")
    else:
        # If the client doesn't accept image, respond with 406 Not Acceptable
        raise HTTPException(status_code=406, detail="Client does not accept an image format")

@app.get("/health")
def health():
    """
    Health check endpoint
    """
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8667)