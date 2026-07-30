import os
import torch


def load_model(model, model_path, device=None):
    """
    Load a trained PyTorch model.

    Args:
        model (torch.nn.Module): Model architecture.
        model_path (str): Path to the saved model.
        device (torch.device, optional): CPU or CUDA device.

    Returns:
        torch.nn.Module: Loaded model.
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    print("=" * 60)
    print("Model loaded successfully!")
    print(f"Model Path : {model_path}")
    print(f"Device     : {device}")
    print("=" * 60)

    return model


if __name__ == "__main__":

    import torch.nn as nn

    model = nn.Linear(10, 2)

    model = load_model(
        model=model,
        model_path="models/test_model.pth"
    )