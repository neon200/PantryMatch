"""
Train a ResNet-based ingredient classifier on the Kaggle Food Ingredient Dataset (51 classes).

Expected directory structure (already in your project):

backend/
  data/
    Train/
      Amaranth/
        Amaranth_1.jpg
        ...
      Apple/
      ...
    val/
      Amaranth/
      Apple/
      ...

This script uses PyTorch + torchvision and transfer learning from a pretrained ResNet18.

Usage (from backend/ directory, inside venv):

  pip install torch torchvision
  python ml_train_ingredients_model.py --epochs 15 --batch-size 32

The best model will be saved as:

  backend/models/ingredients_resnet18.pt
"""

import argparse
import os
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def get_dataloaders(train_dir: str, val_dir: str, batch_size: int = 32, num_workers: int = 2):
    """
    Create PyTorch DataLoaders for train and validation sets using ImageFolder.
    """
    # Standard ImageNet normalization (since we use an ImageNet-pretrained backbone)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, train_dataset.classes


def build_model(num_classes: int):
    """
    Build a ResNet18 model for classification and replace final layer with num_classes outputs.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def train(args):
    backend_dir = Path(__file__).resolve().parent
    data_dir = backend_dir / "data"
    train_dir = data_dir / "Train"
    val_dir = data_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Train/val directories not found. Expected:\n{train_dir}\n{val_dir}"
        )

    print(f"Loading data from:\n  Train: {train_dir}\n  Val:   {val_dir}")
    train_loader, val_loader, class_names = get_dataloaders(
        str(train_dir), str(val_dir), batch_size=args.batch_size, num_workers=args.num_workers
    )
    num_classes = len(class_names)
    print(f"Detected {num_classes} ingredient classes.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Some older torch versions don't support the 'verbose' argument here
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_acc = 0.0
    models_dir = backend_dir / "models"
    models_dir.mkdir(exist_ok=True)
    best_model_path = models_dir / "ingredients_resnet18.pt"
    class_index_path = models_dir / "ingredients_classes.txt"

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                },
                best_model_path,
            )
            with open(class_index_path, "w", encoding="utf-8") as f:
                for name in class_names:
                    f.write(f"{name}\n")
            print(f"Saved new best model to {best_model_path} (val_acc={best_val_acc:.4f})")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model: {best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet ingredient classifier.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader worker processes.",
    )
    args = parser.parse_args()
    train(args)


