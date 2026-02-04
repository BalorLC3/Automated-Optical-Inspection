from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
from PIL import Image
import io
import os
import logging
import base64
import cv2
import numpy as np

app = FastAPI(title="YOLO Inference Service")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger('inference-service')

MODEL_PATH = "runs/detect/steel_defect_26n/weights/best.pt"

print(f"Loading model from: {os.path.abspath(MODEL_PATH)}")

try:
    model = YOLO(MODEL_PATH)
    logger.info(f"Model loaded successfully.")
except Exception as e:
    logger.critical(f"Error loading model: {e}")
    model = None

@app.get("/health")
def health():
    if model is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "ok", "device": str(model.device)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    # Validation
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Read Image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Inference
    results = model.predict(image, conf=0.25)

    # Format Results
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            detections.append({
                "class_name": class_name,
                "class_id": class_id,
                "confidence": round(confidence, 4),
                "bbox": {
                    "x1": round(x1, 2), 
                    "y1": round(y1, 2), 
                    "x2": round(x2, 2), 
                    "y2": round(y2, 2)
                }
            })

    # Draw boxes on image and convert to base64
    processed_image_base64 = draw_boxes_on_image(image, detections)

    return {
        "filename": file.filename,
        "detections": detections,
        "processed_image": processed_image_base64
    }

def draw_boxes_on_image(image, detections):
    """Draw bounding boxes on image and return as base64"""
    # Convert PIL to OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    for det in detections:
        bbox = det['bbox']
        x1, y1 = int(bbox['x1']), int(bbox['y1'])
        x2, y2 = int(bbox['x2']), int(bbox['y2'])
        label = det['class_name']
        conf = det['confidence']
        
        # Draw red rectangle
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 61, 255), 3)
        
        # Label text
        text = f"{label} {conf*100:.1f}%"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Label background
        cv2.rectangle(img_cv, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), (0, 61, 255), -1)
        
        # Label text
        cv2.putText(img_cv, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Convert to base64
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG", quality=95)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return f"data:image/jpeg;base64,{img_base64}"

if __name__ == "__main__": # for development
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)