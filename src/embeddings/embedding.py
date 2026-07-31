import torch
import torch.nn as nn


class Embedding(nn.Module):
    """
    Embedding Layer

    Converts token indices into dense vector representations.

    Input Shape:
        (batch_size, sequence_length)

    Output Shape:
        (batch_size, sequence_length, embedding_dim)
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

    def forward(self, x):
        """
        Forward Pass

        Args:
            x (Tensor):
                Shape -> (batch_size, sequence_length)

        Returns:
            Tensor:
                Shape -> (batch_size, sequence_length, embedding_dim)
        """
        return self.embedding(x)


if __name__ == "__main__":

    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 128

    model = Embedding(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM
    )

    sample_input = torch.randint(
        0,
        VOCAB_SIZE,
        (4, 10)
    )

    output = model(sample_input)

    print("=" * 60)
    print("Embedding Layer")
    print("=" * 60)
    print("Input Shape :", sample_input.shape)
    print("Output Shape:", output.shape)
    print("=" * 60)