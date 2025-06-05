from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, Response
from ultralytics import YOLO
from PIL import Image
import sqlite3
import os
import uuid
import shutil
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from pydantic import BaseModel
from typing import Optional
import tempfile
import logging
import json

# Disable GPU usage
import torch
torch.cuda.is_available = lambda: False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# AWS Configuration
AWS_REGION = "eu-central-1"
S3_BUCKET_NAME = "khaledphotosbuckettelegram"

# Initialize S3 client (using IAM role, no credentials needed)
try:
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    logger.info("S3 client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize S3 client: {e}")
    s3_client = None

# Local directories
UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"
DB_PATH = "predictions.db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

# Download the AI model (tiny model ~6MB)
model = YOLO("yolov8n.pt")

# Pydantic model for request body
class ImageNameRequest(BaseModel):
    image_name: str

# Initialize SQLite
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        # Create the predictions main table to store the prediction session
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_sessions (
                uid TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                original_image TEXT,
                predicted_image TEXT,
                s3_original_key TEXT,
                s3_predicted_key TEXT
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

def save_prediction_session(uid, original_image, predicted_image, s3_original_key=None, s3_predicted_key=None):
    """
    Save prediction session to database
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO prediction_sessions (uid, original_image, predicted_image, s3_original_key, s3_predicted_key)
            VALUES (?, ?, ?, ?, ?)
        """, (uid, original_image, predicted_image, s3_original_key, s3_predicted_key))

def save_detection_object(prediction_uid, label, score, box):
    """
    Save detection object to database
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO detection_objects (prediction_uid, label, score, box)
            VALUES (?, ?, ?, ?)
        """, (prediction_uid, label, score, str(box)))

def download_from_s3(s3_key, local_path):
    """
    Download file from S3 to local path
    """
    if not s3_client:
        raise HTTPException(status_code=500, detail="S3 client not available")
    
    try:
        s3_client.download_file(S3_BUCKET_NAME, s3_key, local_path)
        logger.info(f"Downloaded {s3_key} from S3 to {local_path}")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            logger.error(f"File {s3_key} not found in S3 bucket {S3_BUCKET_NAME}")
            raise HTTPException(status_code=404, detail=f"Image {s3_key} not found in S3")
        else:
            logger.error(f"Failed to download {s3_key} from S3: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to download image from S3: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error downloading {s3_key} from S3: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error downloading image: {str(e)}")

def upload_to_s3(local_path, s3_key):
    """
    Upload file from local path to S3
    """
    if not s3_client:
        raise HTTPException(status_code=500, detail="S3 client not available")
    
    try:
        s3_client.upload_file(local_path, S3_BUCKET_NAME, s3_key)
        logger.info(f"Uploaded {local_path} to S3 as {s3_key}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload {local_path} to S3: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload image to S3: {str(e)}")

def process_image(image_path, uid):
    """
    Process image with YOLO model and return results
    """
    try:
        # Run YOLO prediction
        results = model(image_path, device="cpu")
        
        # Create annotated image
        annotated_frame = results[0].plot()  # NumPy image with boxes
        annotated_image = Image.fromarray(annotated_frame)
        
        # Save predicted image locally
        ext = os.path.splitext(image_path)[1] or '.jpg'
        predicted_path = os.path.join(PREDICTED_DIR, uid + ext)
        annotated_image.save(predicted_path)
        
        return results, predicted_path
    except Exception as e:
        logger.error(f"Failed to process image {image_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

@app.post("/predict")
async def predict(
    image_name_request: Optional[ImageNameRequest] = None,
    file: Optional[UploadFile] = File(None)
):
    """
    Predict objects in an image with fallback logic:
    1. First check for image_name in request body
    2. If not found, check for attached file
    3. Otherwise return 400 bad request
    """
    uid = str(uuid.uuid4())
    s3_original_key = None
    s3_predicted_key = None
    original_path = None
    predicted_path = None
    
    try:
        # Check for image_name in request body first
        if image_name_request and image_name_request.image_name:
            image_name = image_name_request.image_name
            logger.info(f"Processing S3 image: {image_name}")
            
            # Download image from S3
            ext = os.path.splitext(image_name)[1] or '.jpg'
            original_path = os.path.join(UPLOAD_DIR, uid + ext)
            download_from_s3(image_name, original_path)
            s3_original_key = image_name
            
            # Process image
            results, predicted_path = process_image(original_path, uid)
            
            # Upload predicted image to S3
            predicted_s3_key = f"predicted/{uid}{ext}"
            upload_to_s3(predicted_path, predicted_s3_key)
            s3_predicted_key = predicted_s3_key
            
            # Save to database
            save_prediction_session(uid, original_path, predicted_path, s3_original_key, s3_predicted_key)
            
            # Extract detection results
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
                "s3_predicted_key": s3_predicted_key,
                "source": "s3"
            }
        
        # If no image_name found, check for attached file
        elif file and file.filename:
            logger.info(f"Processing uploaded file: {file.filename}")
            
            ext = os.path.splitext(file.filename)[1] or '.jpg'
            original_path = os.path.join(UPLOAD_DIR, uid + ext)
            
            # Save uploaded file
            with open(original_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            # Process image
            results, predicted_path = process_image(original_path, uid)
            
            # Save to database (without S3 keys for uploaded files)
            save_prediction_session(uid, original_path, predicted_path)
            
            # Extract detection results
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
                "source": "upload"
            }
        
        # If neither image_name nor file found, return 400
        else:
            raise HTTPException(
                status_code=400, 
                detail="Bad request: Either provide 'image_name' in request body or attach an image file"
            )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

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
            "s3_original_key": session["s3_original_key"],
            "s3_predicted_key": session["s3_predicted_key"],
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
    return {"status": "ok", "s3_available": s3_client is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8667)