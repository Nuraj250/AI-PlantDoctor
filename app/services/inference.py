import torch
from torchvision import transforms
from PIL import Image
import os
import csv
from datetime import datetime
import json

# Model and classes
MODEL_PATH = os.getenv("MODEL_PATH", "app/models/plant_model.pt")
CLASS_NAMES = ["Healthy", "Leaf Spot", "Rust", "Blight", "Powdery Mildew"]  # Update based on your dataset

TREATMENT_GUIDE_PATH = "app/utils/treatment_guide.json"
with open(TREATMENT_GUIDE_PATH) as f:
    TREATMENT_GUIDE = json.load(f)

# Load model
model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
model.eval()

# Transform image for model input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def log_prediction(label: str, confidence: float):
    os.makedirs("logs", exist_ok=True)
    file_path = "logs/prediction_history.csv"
    log_exists = os.path.exists(file_path)

    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(["timestamp", "label", "confidence"])
        writer.writerow([datetime.now().isoformat(), label, round(confidence, 4)])

def predict_disease(image: Image.Image) -> tuple[str, float]:
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_class = torch.max(probs, 0)

    label = CLASS_NAMES[predicted_class.item()]
    confidence_score = round(confidence.item(), 4)
    treatment = TREATMENT_GUIDE.get(label, "No treatment info available.")

    # ✅ Log the prediction for dashboard
    log_prediction(label, confidence_score)

    return {
        "label": label,
        "confidence": confidence_score,
        "treatment": treatment
    }