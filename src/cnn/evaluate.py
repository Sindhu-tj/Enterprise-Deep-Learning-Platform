import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from model import CNN


# ======================================================
# Device Configuration
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================
# Hyperparameters
# ======================================================
BATCH_SIZE = 64
NUM_CLASSES = 10


# ======================================================
# Dataset
# ======================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

test_dataset = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ======================================================
# Load Model
# ======================================================
model = CNN(num_classes=NUM_CLASSES)

model.load_state_dict(
    torch.load(
        "models/cnn_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()


criterion = nn.CrossEntropyLoss()

all_predictions = []
all_labels = []

running_loss = 0.0


# ======================================================
# Evaluation
# ======================================================
with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# ======================================================
# Metrics
# ======================================================
accuracy = accuracy_score(
    all_labels,
    all_predictions
)

avg_loss = running_loss / len(test_loader)

print("=" * 60)
print("CNN Evaluation Results")
print("=" * 60)

print(f"Test Loss     : {avg_loss:.4f}")
print(f"Accuracy      : {accuracy:.4f}")

print("\nClassification Report\n")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes
    )
)

print("\nConfusion Matrix\n")
print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)

print("=" * 60)