import math
from typing import NamedTuple

import torch
from torch import nn


class LSTMState(NamedTuple):
    """The states of one layer."""

    hidden: torch.Tensor  # [B, H]
    cell: torch.Tensor  # [B, H]


class GateMasks(NamedTuple):
    """Variational dropout masks of one layer, one per row and gate, scaled by `1 / (1 - p)`."""

    inputs: torch.Tensor  # [B, 4, I]
    hidden: torch.Tensor  # [B, 4, H]


class GateDropoutLSTMLayer(nn.Module):
    """LSTM layer with variational dropout on each gate's input and hidden state.

    The masks are passed in rather than drawn here, so the caller decides how long a mask lives:
    one pass in training, one draw for the encoder, one step for the decoder in sampling.
    """

    def __init__(self, *, input_size: int, hidden_size: int, dropout: float) -> None:
        """
        Args:
            input_size: Width of every input.
            hidden_size: Width of the hidden and cell states.
            dropout: Probability of dropping a gate input or hidden feature.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        bound = 1.0 / math.sqrt(hidden_size)
        self.input_weights = nn.Parameter(
            torch.empty(4, hidden_size, input_size).uniform_(-bound, bound)
        )
        self.hidden_weights = nn.Parameter(
            torch.empty(4, hidden_size, hidden_size).uniform_(-bound, bound)
        )
        self.bias = nn.Parameter(torch.empty(4, hidden_size).uniform_(-bound, bound))

    def sample_masks(self, rows: int) -> GateMasks:
        """Draw one Bernoulli mask per row and gate, on the input and on the hidden state."""

        def draw(width: int) -> torch.Tensor:
            keep = torch.rand(size=(rows, 4, width), device=self.bias.device) >= self.dropout
            return keep.to(dtype=self.bias.dtype) / (1.0 - self.dropout)

        return GateMasks(inputs=draw(self.input_size), hidden=draw(self.hidden_size))

    def _step(
        self, projected_inputs: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor
    ) -> LSTMState:
        """One LSTM step from an already projected input and a gate-wise masked hidden state.

        Args:
            projected_inputs: `[B, 4, H]`, the input's contribution to each gate.
            hidden: `[B, 4, H]`, the previous hidden state as each gate reads it.
            cell: `[B, H]`, the previous cell state.

        Returns:
            The next hidden and cell states.
        """
        gates = (
            projected_inputs + torch.einsum('bgk,ghk->bgh', hidden, self.hidden_weights) + self.bias
        )
        input_gate, forget_gate, candidate, output_gate = gates.unbind(dim=1)  # 4 x [B, H]
        cell = forget_gate.sigmoid() * cell + input_gate.sigmoid() * candidate.tanh()
        return LSTMState(hidden=output_gate.sigmoid() * cell.tanh(), cell=cell)

    def forward(
        self,
        inputs: torch.Tensor,
        state: LSTMState,
        *,
        masks: GateMasks,
        advancing: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LSTMState]:
        """Run the layer over inputs `[B, T, I]` from `state`.

        Args:
            inputs: One input per position.
            state: The hidden and cell states before the first position.
            masks: The dropout masks, shared by every position.
            advancing: `[B, T]`, True where a row advances and False where it keeps its state,
                or None to advance every row at every position.

        Returns:
            The hidden state after each position, `[B, T, H]`, and the final state.
        """
        gate_inputs = inputs.unsqueeze(dim=2).expand(-1, -1, 4, -1)  # [B, T, I] -> [B, T, 4, I]
        gate_inputs = gate_inputs * masks.inputs.unsqueeze(dim=1)
        projected = torch.einsum('btgi,ghi->btgh', gate_inputs, self.input_weights)
        hidden, cell = state
        outputs = []
        for position in range(inputs.size(dim=1)):
            gate_hidden = hidden.unsqueeze(dim=1).expand(-1, 4, -1) * masks.hidden  # [B, 4, H]
            next_hidden, next_cell = self._step(
                projected_inputs=projected[:, position], hidden=gate_hidden, cell=cell
            )
            if advancing is not None:
                advance = advancing[:, position].unsqueeze(dim=1)  # [B, 1]
                next_hidden = torch.where(condition=advance, input=next_hidden, other=hidden)
                next_cell = torch.where(condition=advance, input=next_cell, other=cell)
            hidden, cell = next_hidden, next_cell
            outputs.append(hidden)
        # T x [B, H] -> [B, T, H]
        return torch.stack(outputs, dim=1), LSTMState(hidden=hidden, cell=cell)


class GateDropoutStack(nn.Module):
    """Stacked `GateDropoutLSTMLayer` layers, each reading the outputs of the layer below."""

    def __init__(self, *, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        """
        Args:
            input_size: Width of the bottom layer's inputs.
            hidden_size: Width of every layer's states.
            num_layers: Number of layers.
            dropout: Probability of dropping a gate input or hidden feature.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList(
            GateDropoutLSTMLayer(
                input_size=input_size if layer == 0 else hidden_size,
                hidden_size=hidden_size,
                dropout=dropout,
            )
            for layer in range(num_layers)
        )

    def sample_masks(self, rows: int) -> list[GateMasks]:
        """Draw the dropout masks of every layer, bottom first."""
        return [layer.sample_masks(rows) for layer in self.layers]

    def forward_chained(
        self,
        inputs: torch.Tensor,
        state: LSTMState,
        *,
        masks: list[GateMasks],
        advancing: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[LSTMState]]:
        """Run the stack over inputs `[B, T, I]`, the bottom layer from `state` and every upper
        layer from the final state of the layer below.

        Args:
            inputs: One input per position.
            state: The bottom layer's state before the first position.
            masks: The dropout masks of every layer.
            advancing: `[B, T]`, True where a row advances and False where it keeps its state,
                or None to advance every row at every position.

        Returns:
            The top layer's hidden state after each position, `[B, T, H]`, and every layer's
            final state.
        """
        states = []
        for layer, layer_masks in zip(self.layers, masks, strict=True):
            inputs, state = layer(
                inputs=inputs, state=state, masks=layer_masks, advancing=advancing
            )
            states.append(state)
        return inputs, states

    def forward(
        self, inputs: torch.Tensor, states: list[LSTMState], *, masks: list[GateMasks]
    ) -> tuple[torch.Tensor, list[LSTMState]]:
        """Run the stack over inputs `[B, T, I]`, every layer from its own state in `states`.

        Returns:
            The top layer's hidden state after each position, `[B, T, H]`, and every layer's
            final state.
        """
        new_states = []
        for layer, state, layer_masks in zip(self.layers, states, masks, strict=True):
            inputs, state = layer(inputs=inputs, state=state, masks=layer_masks)
            new_states.append(state)
        return inputs, new_states
