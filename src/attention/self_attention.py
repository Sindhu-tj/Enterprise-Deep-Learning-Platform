import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    """
    Self-Attention Layer

    Input Shape:
        (batch_size, sequence_length, embedding_dim)

    Output Shape:
        (batch_size, sequence_length, embedding_dim)
    """

    def __init__(self, embedding_dim):
        super(SelfAttention, self).__init__()

        self.embedding_dim = embedding_dim

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        # Generate Query, Key, Value
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Attention Scores
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.embedding_dim)

        # Attention Weights
        attention_weights = self.softmax(scores)

        # Context Vector
        output = torch.matmul(attention_weights, V)

        return output, attention_weights


if __name__ == "__main__":

    batch_size = 2
    sequence_length = 5
    embedding_dim = 64

    x = torch.randn(batch_size, sequence_length, embedding_dim)

    model = SelfAttention(embedding_dim)

    output, weights = model(x)

    print("=" * 50)
    print("Input Shape:", x.shape)
    print("Output Shape:", output.shape)
    print("Attention Shape:", weights.shape)
    print("=" * 50)