import torch
from torch import nn

from possible_futures.datasets.collation import PrefixTokens
from possible_futures.models.lstm import DropoutMasks, LSTMState, VariationalLSTM


class Encoder(nn.Module):
    """Reads the real events of a prefix with chained layers: each reads the outputs of the layer
    below and starts from its final state."""

    def __init__(self, *, embedding: nn.Embedding, lstm: VariationalLSTM) -> None:
        """
        Args:
            embedding: Embeds every prefix token.
            lstm: The layers reading the embeddings.
        """
        super().__init__()
        self.embedding = embedding
        self.lstm = lstm

    def sample_masks(self, rows: int) -> list[DropoutMasks]:
        """Draw the dropout masks of every layer, bottom first."""
        return self.lstm.sample_masks(rows)

    def forward(self, prefixes: PrefixTokens, *, masks: list[DropoutMasks]) -> LSTMState:
        """Read the real events of the prefixes.

        Args:
            prefixes: The prefix tokens.
            masks: The dropout masks, one per layer.

        Returns:
            LSTMState: The top layer's final state.
        """
        inputs = self.embedding(prefixes.tokens)  # [B, P] -> [B, P, D]
        positions = torch.arange(end=inputs.size(dim=1), device=inputs.device)
        active = positions.unsqueeze(dim=0) < prefixes.lengths.unsqueeze(dim=1)  # [B, P]
        zeros = inputs.new_zeros(size=(inputs.size(dim=0), self.lstm.hidden_size))
        _, states = self.lstm.forward_from_state(
            inputs=inputs,
            state=LSTMState(hidden=zeros, cell=zeros),
            masks=masks,
            active=active,
        )
        return states[-1]
