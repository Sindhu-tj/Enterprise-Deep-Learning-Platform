import torch
from PIL import Image
from torchvision import transforms

from model import CNN


# =====================================================
# Device Configuration
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# Class Labels (CIFAR-10)
# =====================================================
classes = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


# =====================================================
# Image Transform
# =====================================================
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# =====================================================
# Load Model
# =====================================================
model = CNN(num_classes=10)

model.load_state_dict(
    torch.load(
        "models/cnn_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()


# =====================================================
# Prediction Function
# =====================================================
def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image)

        _, predicted = torch.max(outputs, 1)

    return classes[predicted.item()]


# =====================================================
# Test
# =====================================================
if __name__ == "__main__":

    image_path = "sample.jpg"  # Replace with your image

    prediction = predict(image_path)

    print("=" * 50)
    print(f"Predicted Class : {prediction}")
    print("=" * 50)