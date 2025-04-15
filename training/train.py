import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from training.model import build_model
from training.dataset import get_data_loaders
import os

def train_model(data_dir: str, epochs: int = 10, lr: float = 1e-3, batch_size: int = 32, save_path: str = "app/models/plant_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_names = get_data_loaders(data_dir, batch_size)

    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {running_loss / len(train_loader):.4f}")

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model, save_path)
    print(f"Model saved to {save_path}")
