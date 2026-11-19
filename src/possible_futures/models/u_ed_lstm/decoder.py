import torch
from jaxtyping import Float, Int64
from torch import nn

from possible_futures.models.lstm import DropoutMasks, LSTMState, VariationalLSTM
from possible_futures.models.u_ed_lstm.logits import GaussianHead, GaussianLogits


class Decoder(nn.Module):
    """Predicts the logit distribution of the next event from the previous one.

    At its first step the bottom layer starts from the encoder's final state and each upper layer
    from the state the layer below just reached; from then on every layer carries its own state.
    """

    def __init__(
        self, *, embedding: nn.Embedding, lstm: VariationalLSTM, head: GaussianHead
    ) -> None:
        """
        Args:
            embedding: Embeds every input token.
            lstm: The layers reading the embeddings.
            head: Maps the top layer's hidden state to the logits of the next event.
        """
        super().__init__()
        self.embedding = embedding
        self.lstm = lstm
        self.head = head

    def sample_masks(self, rows: int) -> list[DropoutMasks]:
        """Draw the dropout masks of every layer, bottom first."""
        return self.lstm.sample_masks(rows)

    def start(
        self, tokens: Int64[torch.Tensor, ' B'], encoded: LSTMState, *, masks: list[DropoutMasks]
    ) -> tuple[GaussianLogits, list[LSTMState]]:
        """Read the first input of every row, starting from the encoder's final state.

        Args:
            tokens: `[B]` the last prefix event of every row.
            encoded: The encoder's final state.
            masks: The dropout masks, one per layer.

        Returns:
            The logits `[B, C]` of the next event, and each layer's new state.
        """
        outputs, states = self.lstm.forward_from_state(
            inputs=self._read(tokens), state=encoded, masks=masks
        )
        return self.head(outputs.squeeze(dim=1)), states  # [B, 1, H] -> [B, H]

    def step(
        self,
        tokens: Int64[torch.Tensor, ' B'],
        states: list[LSTMState],
        *,
        masks: list[DropoutMasks],
    ) -> tuple[GaussianLogits, list[LSTMState]]:
        """Read one input per row after the first, every layer from its own state.

        Args:
            tokens: `[B]` the previous event of every row.
            states: Each layer's state.
            masks: The dropout masks, one per layer.

        Returns:
            The logits `[B, C]` of the next event, and each layer's new state.
        """
        outputs, states = self.lstm(inputs=self._read(tokens), states=states, masks=masks)
        return self.head(outputs.squeeze(dim=1)), states  # [B, 1, H] -> [B, H]

    def _read(self, tokens: Int64[torch.Tensor, ' B']) -> Float[torch.Tensor, 'B 1 D']:
        """The embedding of every token, as a sequence of one input."""
        return self.embedding(tokens).unsqueeze(dim=1)  # [B] -> [B, 1, D]
