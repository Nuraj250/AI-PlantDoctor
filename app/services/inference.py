import torch
from torchvision import transforms
from PIL import Image
import os

MODEL_PATH = os.getenv("MODEL_PATH", "app/models/plant_model.pt")
CLASS_NAMES = ["Healthy", "Powdery Mildew", "Leaf Spot", "Rust", "Blight"]  # Example classes

# Load once when module is imported
model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
model.eval()

# Transform image for model input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_disease(image: Image.Image) -> tuple[str, float]:
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_class = torch.max(probs, 0)
    return CLASS_NAMES[predicted_class.item()], round(confidence.item(), 4)
