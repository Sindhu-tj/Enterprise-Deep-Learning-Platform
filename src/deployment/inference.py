import torch


def inference(model, inputs, device=None):
    """
    Perform inference using a trained PyTorch model.

    Args:
        model (torch.nn.Module): Loaded PyTorch model.
        inputs (torch.Tensor): Input tensor.
        device (torch.device, optional): CPU or CUDA device.

    Returns:
        torch.Tensor: Model predictions.
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model.to(device)
    model.eval()

    inputs = inputs.to(device)

    with torch.no_grad():

        outputs = model(inputs)

        if outputs.ndim > 1:
            predictions = torch.argmax(outputs, dim=1)
        else:
            predictions = outputs

    return predictions


if __name__ == "__main__":

    import torch.nn as nn

    model = nn.Linear(10, 3)

    sample = torch.randn(4, 10)

    predictions = inference(
        model=model,
        inputs=sample
    )

    print("=" * 60)
    print("Inference Completed")
    print("=" * 60)
    print("Predictions:")
    print(predictions)
    print("=" * 60)