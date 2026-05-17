import os
import time
import base64

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # img = cv2.flip(img, 1)
    img2 = img.copy()
    # img = cv2.resize(img, (64, 64))

    results = model(img)

    label = results[0].names[results[0].probs.top1]
    confidence = results[0].probs.top1conf.item()

    # Create folder
    os.makedirs("captures/original", exist_ok=True)
    os.makedirs("captures/result", exist_ok=True)

    # Save original
    cv2.imwrite(f"captures/original/original_{int(time.time())}.jpg", img)

    # Save annotated
    annotated = results[0].plot()
    cv2.imwrite(f"captures/result/result_{int(time.time())}.jpg", annotated)

    return {
        "prediction": label,
        "confidence": confidence
    }