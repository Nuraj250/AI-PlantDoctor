from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple, List

def get_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Loads training and validation datasets from directory.

    Args:
        data_dir (str): Path to the root directory containing `train/` and `val/`.
        batch_size (int): Number of images per batch.
        image_size (int): Size to which images will be resized.

    Returns:
        Tuple containing:
            - train_loader (DataLoader)
            - val_loader (DataLoader)
            - class_names (List[str])
    """
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=train_transforms)
    val_dataset = datasets.ImageFolder(f"{data_dir}/val", transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_dataset.classes
