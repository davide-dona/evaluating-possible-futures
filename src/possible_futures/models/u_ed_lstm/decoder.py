import torch
from jaxtyping import Float, Int64
from torch import nn

from possible_futures.models.lstm import DropoutMasks, LSTMState, VariationalLSTM
from possible_futures.models.u_ed_lstm.logits import GaussianHead, GaussianLogits


class Decoder(nn.Module):
    """Writes a suffix one token at a time: reads the previous token and predicts the Gaussian
    logits of the next.

    The first step starts from the encoder's summary of the prefix (`start`); every later step
    continues from the decoder's own state (`step`).
    """

    def __init__(
        self, *, embedding: nn.Embedding, lstm: VariationalLSTM, head: GaussianHead
    ) -> None:
        """
        Args:
            embedding: Turns every input token into a vector.
            lstm: The layers reading the embeddings.
            head: Maps the top layer's hidden state to the logits of the next token.
        """
        super().__init__()
        self.embedding = embedding
        self.lstm = lstm
        self.head = head

    def start(
        self, tokens: Int64[torch.Tensor, ' B'], encoded: LSTMState, *, masks: list[DropoutMasks]
    ) -> tuple[GaussianLogits, list[LSTMState]]:
        """Take the first decoding step, from the encoder's summary of the prefix.

        Args:
            tokens: The last prefix token of every row.
            encoded: The encoder's final state.
            masks: The dropout masks, one per layer.

        Returns:
            The logits `[B, V]` of the first suffix token, and each layer's new state.
        """
        outputs, states = self.lstm.forward_from_state(
            inputs=self._embed(tokens), state=encoded, masks=masks
        )
        return self.head(outputs.squeeze(dim=1)), states  # [B, 1, H] -> [B, H]

    def step(
        self,
        tokens: Int64[torch.Tensor, ' B'],
        states: list[LSTMState],
        *,
        masks: list[DropoutMasks],
    ) -> tuple[GaussianLogits, list[LSTMState]]:
        """Take one decoding step after the first, every layer from its own state.

        Args:
            tokens: The previous token of every row.
            states: Each layer's state.
            masks: The dropout masks, one per layer.

        Returns:
            The logits `[B, V]` of the next token, and each layer's new state.
        """
        outputs, states = self.lstm(inputs=self._embed(tokens), states=states, masks=masks)
        return self.head(outputs.squeeze(dim=1)), states  # [B, 1, H] -> [B, H]

    def _embed(self, tokens: Int64[torch.Tensor, ' B']) -> Float[torch.Tensor, 'B 1 D']:
        """Embed one token per row, as a sequence of one input.

        Args:
            tokens: The token of every row.

        Returns:
            torch.Tensor: The embedding of every token.
        """
        return self.embedding(tokens).unsqueeze(dim=1)  # [B] -> [B, 1, D]
