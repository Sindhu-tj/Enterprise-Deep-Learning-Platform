import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from .model import CNN


# ==========================================================
# Configuration
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
NUM_CLASSES = 10

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.pth")


# ==========================================================
# Data Loader
# ==========================================================

def get_train_loader():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )


# ==========================================================
# Train Function
# ==========================================================

def train():

    train_loader = get_train_loader()

    model = CNN(num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("=" * 60)
    print("Training CNN...")
    print("=" * 60)

    for epoch in range(NUM_EPOCHS):

        model.train()

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Loss: {epoch_loss:.4f}"
        )

    os.makedirs(MODEL_DIR, exist_ok=True)

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print("\nModel saved successfully.")
    print(MODEL_PATH)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    train()