import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import csv

from training.model import build_model
from training.dataset import get_data_loaders

plt.ion()  # Enable interactive plot mode (for Windows too)

def train_model(data_dir: str, epochs: int = 10, lr: float = 1e-3, batch_size: int = 32, save_path: str = "app/models/plant_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_names = get_data_loaders(data_dir, batch_size)
    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # Early stopping and metrics
    best_val_loss = float('inf')
    patience = 3
    trigger_times = 0
    losses = []

    # CSV logging (optional)
    log_file = "training_log.csv"
    log_exists = os.path.exists(log_file)
    with open(log_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not log_exists:
            writer.writerow(["epoch", "train_loss", "val_loss", "val_acc"])

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

        avg_train_loss = running_loss / len(train_loader)
        losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        print(f"✅ Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Val Acc={val_acc:.4f}")

        # Save to CSV log
        with open(log_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch + 1, round(avg_train_loss, 4), round(avg_val_loss, 4), round(val_acc, 4)])

        # Live plot
        plt.clf()
        plt.plot(losses, marker='o', label="Train Loss")
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.pause(0.1)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            torch.save(model, save_path)
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("🛑 Early stopping triggered.")
                break

    plt.ioff()
    plt.savefig("training_loss.png")
    print(f"\n✅ Model saved to: {save_path}")
    print("📈 Training loss plot saved to training_loss.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Root folder with 'train' and 'val' subfolders")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="app/models/plant_model.pt")
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        save_path=args.save_path
    )
