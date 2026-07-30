import os
import torch


def save_model(model, model_path):
    """
    Save a PyTorch model.

    Args:
        model (torch.nn.Module): Trained model.
        model_path (str): Path to save the model.
    """

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    torch.save(model.state_dict(), model_path)

    print("=" * 60)
    print("Model saved successfully!")
    print(f"Location : {model_path}")
    print("=" * 60)


if __name__ == "__main__":

    import torch.nn as nn

    model = nn.Linear(10, 2)

    save_model(
        model=model,
        model_path="models/test_model.pth"
    )