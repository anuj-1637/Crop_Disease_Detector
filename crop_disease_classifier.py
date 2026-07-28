

import os
import shutil
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = "dataset"          
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
IMG_SIZE = 224
MODEL_SAVE_PATH = "crop_disease_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def split_dataset(source_dir, dest_dir, val_ratio=0.2):
    """
    Splits a flat dataset (class folders directly containing images)
    into train/ and val/ folders.
    source_dir: original PlantVillage folder (class subfolders)
    dest_dir: where to create train/ and val/
    """
    classes = os.listdir(source_dir)
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        images = os.listdir(cls_path)
        random.shuffle(images)
        val_count = int(len(images) * val_ratio)
        val_images = images[:val_count]
        train_images = images[val_count:]

        for split_name, split_images in [("train", train_images), ("val", val_images)]:
            split_class_dir = os.path.join(dest_dir, split_name, cls)
            os.makedirs(split_class_dir, exist_ok=True)
            for img_name in split_images:
                src = os.path.join(cls_path, img_name)
                dst = os.path.join(split_class_dir, img_name)
                shutil.copy(src, dst)
    print("Dataset split complete.")



train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])  
])


val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])



def get_dataloaders():
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    class_names = train_dataset.classes
    print(f"Found {len(class_names)} classes: {class_names}")
    print(f"Train images: {len(train_dataset)}, Val images: {len(val_dataset)}")

    return train_loader, val_loader, class_names



def build_model(num_classes):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    return model.to(DEVICE)



def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Accuracy: {val_acc*100:.2f}%")

    return history


def evaluate_model(model, val_loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved as confusion_matrix.png")


def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_acc"], label="Val Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Validation Accuracy")

    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("Training curves saved as training_curves.png")


def predict_image(model, image_path, class_names):
    from PIL import Image
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image = val_transforms(image).unsqueeze(0).to(DEVICE)  # add batch dimension

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    predicted_class = class_names[predicted.item()]
    confidence_pct = confidence.item() * 100
    return predicted_class, confidence_pct



if __name__ == "__main__":


    train_loader, val_loader, class_names = get_dataloaders()

    model = build_model(num_classes=len(class_names))

    history = train_model(model, train_loader, val_loader)

    evaluate_model(model, val_loader, class_names)
    plot_history(history)

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names
    }, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

