import math
from typing import NamedTuple

import torch
from torch import nn

# The hidden and cell states of one layer, `[B, H]` each.
LSTMState = tuple[torch.Tensor, torch.Tensor]


class GateMasks(NamedTuple):
    """Variational dropout masks of one layer, one per row and gate, scaled by `1 / (1 - p)`."""

    inputs: torch.Tensor  # [B, 4, I]
    hidden: torch.Tensor  # [B, 4, H]

    def select(self, index: torch.Tensor) -> 'GateMasks':
        """Keep the masks of the rows at `index`."""
        return GateMasks(inputs=self.inputs[index], hidden=self.hidden[index])


def _lstm_step(
    projected_inputs: torch.Tensor,
    hidden: torch.Tensor,
    cell: torch.Tensor,
    hidden_weights: torch.Tensor,
    bias: torch.Tensor,
) -> LSTMState:
    """One LSTM step from an already projected input and a gate-wise masked hidden state.

    Args:
        projected_inputs: `[B, 4, H]`, the input's contribution to each gate.
        hidden: `[B, 4, H]`, the previous hidden state as each gate reads it.
        cell: `[B, H]`, the previous cell state.
        hidden_weights: `[4, H, H]`.
        bias: `[4, H]`.

    Returns:
        The next hidden and cell states.
    """
    gates = projected_inputs + torch.einsum('bgk,ghk->bgh', hidden, hidden_weights) + bias
    input_gate, forget_gate, candidate, output_gate = gates.unbind(dim=1)  # 4 x [B, H]
    cell = forget_gate.sigmoid() * cell + input_gate.sigmoid() * candidate.tanh()
    return output_gate.sigmoid() * cell.tanh(), cell


class GateDropoutLSTMCell(nn.Module):
    """LSTM layer with variational dropout on each gate's input and hidden state.

    The masks are passed in rather than drawn here, so the caller decides how long a mask lives:
    one pass in training, one draw for the encoder, one step for the decoder in sampling.
    """

    def __init__(self, *, input_size: int, hidden_size: int) -> None:
        super().__init__()
        bound = 1.0 / math.sqrt(hidden_size)
        self.input_weights = nn.Parameter(
            torch.empty(4, hidden_size, input_size).uniform_(-bound, bound)
        )
        self.hidden_weights = nn.Parameter(
            torch.empty(4, hidden_size, hidden_size).uniform_(-bound, bound)
        )
        self.bias = nn.Parameter(torch.empty(4, hidden_size).uniform_(-bound, bound))

    def sample_masks(self, *, rows: int, p: float, device: torch.device) -> GateMasks:
        """Draw one Bernoulli mask per row and gate, on the input and on the hidden state."""
        input_size, hidden_size = self.input_weights.size(dim=2), self.hidden_weights.size(dim=1)

        def draw(width: int) -> torch.Tensor:
            keep = torch.rand(size=(rows, 4, width), device=device) >= p
            return keep.float() / (1.0 - p)

        return GateMasks(inputs=draw(input_size), hidden=draw(hidden_size))

    def forward(
        self,
        inputs: torch.Tensor,
        state: LSTMState,
        *,
        masks: GateMasks | None,
        keep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LSTMState]:
        """Run the layer over inputs `[B, T, I]` from `state`.

        Args:
            inputs: One input per position.
            state: The hidden and cell states before the first position.
            masks: The dropout masks, shared by every position, or None for no dropout.
            keep: `[B, T]`, False where a row keeps its state instead of advancing, or None to
                advance every row at every position.

        Returns:
            The hidden state after each position, `[B, T, H]`, and the final state.
        """
        gate_inputs = inputs.unsqueeze(dim=2).expand(-1, -1, 4, -1)  # [B, T, I] -> [B, T, 4, I]
        if masks is not None:
            gate_inputs = gate_inputs * masks.inputs.unsqueeze(dim=1)
        projected = torch.einsum('btgi,ghi->btgh', gate_inputs, self.input_weights)
        hidden, cell = state
        outputs = []
        for position in range(inputs.size(dim=1)):
            gate_hidden = hidden.unsqueeze(dim=1).expand(-1, 4, -1)  # [B, H] -> [B, 4, H]
            if masks is not None:
                gate_hidden = gate_hidden * masks.hidden
            next_hidden, next_cell = _lstm_step(
                projected_inputs=projected[:, position],
                hidden=gate_hidden,
                cell=cell,
                hidden_weights=self.hidden_weights,
                bias=self.bias,
            )
            if keep is not None:
                advance = keep[:, position].unsqueeze(dim=1)  # [B, 1]
                next_hidden = torch.where(advance, next_hidden, hidden)
                next_cell = torch.where(advance, next_cell, cell)
            hidden, cell = next_hidden, next_cell
            outputs.append(hidden)
        return torch.stack(outputs, dim=1), (hidden, cell)  # T x [B, H] -> [B, T, H]
