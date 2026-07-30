import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Layer

    Input Shape:
        (batch_size, sequence_length, embedding_dim)

    Output Shape:
        (batch_size, sequence_length, embedding_dim)
    """

    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self).__init__()

        assert embedding_dim % num_heads == 0, \
            "Embedding dimension must be divisible by number of heads."

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        self.fc_out = nn.Linear(embedding_dim, embedding_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        batch_size = x.size(0)
        seq_len = x.size(1)

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        attention_weights = self.softmax(scores)

        out = torch.matmul(attention_weights, V)

        out = out.transpose(1, 2).contiguous()

        out = out.view(batch_size, seq_len, self.embedding_dim)

        out = self.fc_out(out)

        return out, attention_weights


if __name__ == "__main__":

    batch_size = 2
    sequence_length = 10
    embedding_dim = 128
    num_heads = 8

    x = torch.randn(batch_size, sequence_length, embedding_dim)

    model = MultiHeadAttention(
        embedding_dim=embedding_dim,
        num_heads=num_heads
    )

    output, attention = model(x)

    print("=" * 50)
    print("Input Shape      :", x.shape)
    print("Output Shape     :", output.shape)
    print("Attention Shape  :", attention.shape)
    print("=" * 50)