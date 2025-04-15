from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.services.inference import predict_disease
from PIL import Image
import io

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
