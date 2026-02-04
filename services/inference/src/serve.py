from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
from PIL import Image
import io
import os
import logging 

app = FastAPI(title="YOLO Inference Service")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger('inference-service')

# inside the container, we will map the volume to /app/runs
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
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Read Image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Inference
    results = model.predict(image, conf=0.25)

    # Format Results for Go Backend
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

    return {"filename": file.filename, "detections": detections}

# for debugging
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)