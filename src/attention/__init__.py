import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Basic Additive Attention (Bahdanau Attention)

    Input:
        query  : (batch_size, hidden_dim)
        values : (batch_size, seq_len, hidden_dim)

    Output:
        context_vector : (batch_size, hidden_dim)
        attention_weights : (batch_size, seq_len)
    """

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()

        self.W_query = nn.Linear(hidden_dim, hidden_dim)
        self.W_values = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, query, values):

        # Expand query for each time step
        query = query.unsqueeze(1)

        # Compute alignment scores
        score = self.V(
            torch.tanh(
                self.W_query(query) +
                self.W_values(values)
            )
        )

        # Remove last dimension
        score = score.squeeze(-1)

        # Normalize scores
        attention_weights = F.softmax(score, dim=1)

        # Compute context vector
        context_vector = torch.sum(
            attention_weights.unsqueeze(-1) * values,
            dim=1
        )

        return context_vector, attention_weights


if __name__ == "__main__":

    batch_size = 4
    seq_len = 10
    hidden_dim = 128

    query = torch.randn(batch_size, hidden_dim)
    values = torch.randn(batch_size, seq_len, hidden_dim)

    attention = Attention(hidden_dim)

    context, weights = attention(query, values)

    print("=" * 50)
    print("Context Shape :", context.shape)
    print("Weights Shape :", weights.shape)
    print("=" * 50)