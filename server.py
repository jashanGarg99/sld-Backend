import os
import numpy as np
import cv2

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
@limiter.limit("10/minute")
def home(request: Request):
    return {"message": "ASL API Running"}

# ✅ Prediction route
@app.post("/predict")
@limiter.limit("5/second;30/minute")
async def predict(request: Request, file: UploadFile = File(...)):
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