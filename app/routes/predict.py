from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.services.inference import predict_disease
from PIL import Image
import io
import csv
import os
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        label, confidence = predict_disease(image)
        return JSONResponse(content={"prediction": label, "confidence": confidence})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/training-log")
async def get_training_log():
    log_path = "training_log.csv"
    if not os.path.exists(log_path):
        return JSONResponse(content=[], status_code=200)

    with open(log_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        data = [{
            "epoch": int(row["epoch"]),
            "train_loss": float(row["train_loss"]),
            "val_loss": float(row["val_loss"]),
            "val_acc": float(row["val_acc"])
        } for row in reader]

    return JSONResponse(content=data)

@router.get("/prediction-history")
async def get_prediction_history():
    file_path = "logs/prediction_history.csv"
    if not os.path.exists(file_path):
        return JSONResponse(content=[], status_code=200)

    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)

    return JSONResponse(content=data)