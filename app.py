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

# Initialize SQLite
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

# S3 Configuration - Using instance credentials
AWS_REGION = "eu-central-1"
S3_BUCKET_NAME = "khaledphotosbuckettelegram"

# Initialize S3 client (uses instance IAM role)
s3 = boto3.client("s3", region_name=AWS_REGION)

def download_image_from_s3(image_name: str, local_path: str) -> bool:
    try:
        print(f"Attempting to download from S3: bucket={S3_BUCKET_NAME}, key={image_name}")
        s3.download_file(S3_BUCKET_NAME, image_name, local_path)
        print(f"Successfully downloaded {image_name} from S3")
        return True
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        return False

def upload_image_to_s3(local_path: str, image_name: str) -> bool:
    try:
        s3.upload_file(local_path, S3_BUCKET_NAME, image_name)
        print(f"Successfully uploaded {image_name} to S3")
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
    Handles both JSON requests with image_name and file uploads with fallback logic:
    1. First check for image_name in JSON body (S3 download)
    2. Then check for file upload
    3. Return 400 if neither found
    """
    # Generate unique ID for this prediction
    uid = str(uuid.uuid4())
    
    # Check content type to determine processing method
    content_type = request.headers.get('content-type', '').lower()
    
    # FALLBACK LOGIC: First try JSON with image_name, then file upload
    try:
        # Method 1: Try JSON request with image_name (from S3)
        if 'application/json' in content_type:
            print("Processing JSON request...")
            json_data = await request.json()
            image_name = json_data.get("image_name")
            
            if image_name:
                print(f"Found image_name in JSON: {image_name}")
                return await process_s3_image(uid, image_name)
            else:
                raise HTTPException(status_code=400, detail="image_name field is required in JSON body")
        
        # Method 2: Try multipart file upload
        elif 'multipart/form-data' in content_type:
            print("Processing file upload...")
            form_data = await request.form()
            
            if 'file' in form_data:
                file = form_data['file']
                if file.filename:
                    print(f"Found uploaded file: {file.filename}")
                    return await process_uploaded_file(uid, file)
            
            # If no file found, return error
            raise HTTPException(status_code=400, detail="No file found in form data")
        
        else:
            # Try to read as JSON anyway (some clients may not set correct content-type)
            try:
                json_data = await request.json()
                image_name = json_data.get("image_name")
                
                if image_name:
                    print(f"Found image_name in body (no content-type): {image_name}")
                    return await process_s3_image(uid, image_name)
            except:
                pass
            
            # If nothing works, return error
            raise HTTPException(
                status_code=400, 
                detail="Request must contain either 'image_name' in JSON body or a file upload"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Request processing failed: {str(e)}")

async def process_s3_image(uid: str, image_name: str):
    """Process image downloaded from S3"""
    try:
        # Determine file extension
        ext = os.path.splitext(image_name)[1] or ".jpg"
        original_filename = f"{uid}_original{ext}"
        original_path = os.path.join(UPLOAD_DIR, original_filename)
        
        # Download from S3
        print(f"Downloading image from S3: {image_name}")
        if not download_image_from_s3(image_name, original_path):
            raise HTTPException(status_code=404, detail=f"Image '{image_name}' not found in S3 bucket")
        
        # Verify file was downloaded and is valid
        if not os.path.exists(original_path):
            raise HTTPException(status_code=500, detail="Failed to download image from S3")
        
        return await run_prediction(uid, original_path, ext, f"s3:{image_name}")
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing S3 image: {e}")
        raise HTTPException(status_code=500, detail=f"S3 image processing failed: {str(e)}")

async def process_uploaded_file(uid: str, file):
    """Process uploaded file"""
    try:
        # Determine file extension
        ext = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
        original_filename = f"{uid}_original{ext}"
        original_path = os.path.join(UPLOAD_DIR, original_filename)
        
        # Save uploaded file
        print(f"Saving uploaded file: {file.filename}")
        with open(original_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        return await run_prediction(uid, original_path, ext, f"upload:{file.filename}")
        
    except Exception as e:
        print(f"Error processing uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"File upload processing failed: {str(e)}")

async def run_prediction(uid: str, original_path: str, ext: str, source_info: str):
    """
    Common prediction processing for both S3 and file upload paths
    """
    try:
        predicted_filename = f"{uid}_predicted{ext}"
        predicted_path = os.path.join(PREDICTED_DIR, predicted_filename)

        print(f"Running YOLO prediction on: {original_path} (from {source_info})")
        
        # Verify image file exists and is readable
        if not os.path.exists(original_path):
            raise HTTPException(status_code=500, detail="Image file not found for prediction")
        
        # Run YOLO prediction
        results = model(original_path, device="cpu")
        
        if not results or len(results) == 0:
            raise HTTPException(status_code=500, detail="YOLO prediction returned no results")

        # Create annotated image
        annotated_frame = results[0].plot()
        annotated_image = Image.fromarray(annotated_frame)
        annotated_image.save(predicted_path)

        # Upload both images to S3
        original_s3_key = f"predictions/original_{os.path.basename(original_path)}"
        predicted_s3_key = f"predictions/predicted_{predicted_filename}"
        
        upload_success_original = upload_image_to_s3(original_path, original_s3_key)
        upload_success_predicted = upload_image_to_s3(predicted_path, predicted_s3_key)
        
        if not upload_success_original or not upload_success_predicted:
            print("Warning: Failed to upload some images to S3")

        # Save to database
        save_prediction_session(uid, original_s3_key, predicted_s3_key)

        # Process detection results
        detected_labels = []
        detection_details = []
        
        for box in results[0].boxes:
            label_idx = int(box.cls[0].item())
            label = model.names[label_idx]
            score = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            
            save_detection_object(uid, label, score, bbox)
            detected_labels.append(label)
            detection_details.append({
                "label": label,
                "confidence": round(score, 3),
                "bbox": [round(x, 2) for x in bbox]
            })

        # Clean up local files to save space
        try:
            os.remove(original_path)
            os.remove(predicted_path)
        except:
            pass

        unique_labels = list(set(detected_labels))
        response_data = {
            "success": True,
            "prediction_uid": uid,
            "source": source_info,
            "detection_count": len(results[0].boxes),
            "unique_labels": unique_labels,
            "all_detections": detection_details,
            "original_image_s3_key": original_s3_key,
            "predicted_image_s3_key": predicted_s3_key,
            "message": f"Successfully detected {len(detected_labels)} objects: {', '.join(unique_labels)}"
        }
        
        print(f"Prediction successful: {len(detected_labels)} objects detected")
        return JSONResponse(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# Get prediction history
@app.get("/predictions/{uid}")
async def get_prediction(uid: str):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("""
                SELECT uid, timestamp, original_image, predicted_image
                FROM prediction_sessions WHERE uid = ?
            """, (uid,))
            session = cursor.fetchone()
            
            if not session:
                raise HTTPException(status_code=404, detail="Prediction not found")
            
            cursor = conn.execute("""
                SELECT label, score, box FROM detection_objects 
                WHERE prediction_uid = ? ORDER BY score DESC
            """, (uid,))
            detections = cursor.fetchall()
            
            return {
                "uid": session[0],
                "timestamp": session[1],
                "original_image": session[2],
                "predicted_image": session[3],
                "detections": [
                    {"label": d[0], "score": d[1], "box": d[2]} 
                    for d in detections
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8667)