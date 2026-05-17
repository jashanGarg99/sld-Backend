import os
import numpy as np
import cv2

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change later in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load model
model = YOLO("./best.pt")

# ✅ Health route (important for Render)
@app.get("/")
def home():
    return {"message": "ASL API Running"}

# ✅ Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image"}

        # Optional mirror
        # img = cv2.flip(img, 1)

        # Optional resize
        # img = cv2.resize(img, (96, 96))

        results = model(img)

        label = results[0].names[results[0].probs.top1]
        confidence = float(results[0].probs.top1conf.item())

        return {
            "prediction": label,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}