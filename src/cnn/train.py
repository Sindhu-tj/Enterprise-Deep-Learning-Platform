import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CNN


# =====================================================
# Device Configuration
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# Hyperparameters
# =====================================================
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
NUM_CLASSES = 10


# =====================================================
# Dataset
# =====================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

train_dataset = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# =====================================================
# Model
# =====================================================
model = CNN(num_classes=NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =====================================================
# Training Loop
# =====================================================
print("Training Started...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {avg_loss:.4f}"
    )


# =====================================================
# Save Model
# =====================================================
os.makedirs("models", exist_ok=True)

torch.save(
    model.state_dict(),
    "models/cnn_model.pth"
)

print("\nModel Saved Successfully!")