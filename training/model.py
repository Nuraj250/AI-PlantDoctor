import torch.nn as nn
from torchvision import models

def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False  # Freeze base layers

    # Replace FC layer
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    return model
