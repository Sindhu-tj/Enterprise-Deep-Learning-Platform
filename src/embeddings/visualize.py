import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def visualize_embeddings(embedding_weights, labels=None):
    """
    Visualize embedding vectors using t-SNE.

    Args:
        embedding_weights (torch.Tensor or np.ndarray):
            Shape -> (vocab_size, embedding_dim)

        labels (list, optional):
            Token labels corresponding to each embedding.
    """

    # Convert tensor to numpy
    if hasattr(embedding_weights, "detach"):
        embedding_weights = (
            embedding_weights.detach()
            .cpu()
            .numpy()
        )

    # Reduce dimensions to 2D
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=5
    )

    reduced_embeddings = tsne.fit_transform(
        embedding_weights
    )

    plt.figure(figsize=(10, 8))

    plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        alpha=0.7
    )

    if labels is not None:

        for i, label in enumerate(labels):

            plt.annotate(
                label,
                (
                    reduced_embeddings[i, 0],
                    reduced_embeddings[i, 1]
                ),
                fontsize=9
            )

    plt.title("Embedding Visualization (t-SNE)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":

    import torch

    vocab_size = 20
    embedding_dim = 16

    embeddings = torch.randn(
        vocab_size,
        embedding_dim
    )

    labels = [
        f"word_{i}"
        for i in range(vocab_size)
    ]

    visualize_embeddings(
        embeddings,
        labels
    )