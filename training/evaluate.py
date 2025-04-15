import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from training.dataset import get_data_loaders
from app.services.inference import transform  # reuse transform logic
from torchvision import datasets
import os

def evaluate_model(model_path: str, data_dir: str, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = datasets.ImageFolder(f"{data_dir}/val", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = torch.load(model_path, map_location=device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=val_dataset.classes))

    print("\n🧩 Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
