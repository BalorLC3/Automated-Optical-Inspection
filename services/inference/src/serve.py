from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
from PIL import Image
import io
import torch

app = FastAPI(title="YOLOv10 Inference Service")

# Load model once at startup
# Path assumes you run this from services/inference-py/
# Point this to your trained weights: "runs/steel_defect_v10/weights/best.pt"
MODEL_PATH = "yolov10n.pt" 
try:
    model = YOLO(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

@app.get("/health")
def health():
    return {"status": "ok", "device": str(model.device)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Validation
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type")

    # 2. Read Image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # 3. Inference
    # conf=0.25 is a standard confidence threshold
    results = model.predict(image, conf=0.25)

    # 4. Format Results for Go Backend
    detections = []
    for result in results:
        for box in result.boxes:
            # YOLOv10 outputs: [x1, y1, x2, y2, confidence, class_id]
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

if __name__ == "__main__":
    import uvicorn
    # Run on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)