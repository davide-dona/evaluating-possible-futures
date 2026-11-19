import math
from typing import NamedTuple

import torch
from jaxtyping import Bool, Float, Int64
from torch import nn


class LSTMState(NamedTuple):
    """The memory of one LSTM layer for every row of a batch.

    Attributes:
        hidden: The hidden state, which the layer outputs and the next step reads.
        cell: The cell state, the layer's long-term memory.
    """

    hidden: Float[torch.Tensor, 'B H']
    cell: Float[torch.Tensor, 'B H']

    def select(self, rows: Int64[torch.Tensor, ' R']) -> 'LSTMState':
        """Keep only some rows, such as the ones still decoding.

        Args:
            rows: The indices of the rows kept, in the order kept.

        Returns:
            LSTMState: The states `[R, H]` of the rows kept.
        """
        return LSTMState(hidden=self.hidden[rows], cell=self.cell[rows])


class DropoutMasks(NamedTuple):
    """The variational dropout masks of one layer. Each row has its own masks, one per gate,
    which every step of a sequence reuses. A kept feature is scaled by `1 / (1 - p)`, a dropped
    one is 0.

    Attributes:
        input: The mask on the input each gate reads.
        hidden: The mask on the previous hidden state each gate reads.
    """

    input: Float[torch.Tensor, 'B 4 I']
    hidden: Float[torch.Tensor, 'B 4 H']


class VariationalLSTMLayer(nn.Module):
    """LSTM layer with variational dropout: each gate sees its input and the previous hidden
    state through its own dropout mask, fixed for a whole sequence."""

    def __init__(self, *, input_size: int, hidden_size: int, dropout: float) -> None:
        """
        Args:
            input_size: Width of every input, `I`.
            hidden_size: Width of the hidden and cell states, `H`.
            dropout: Probability of dropping a gate input or hidden feature.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        # The default initialization of torch.nn.LSTM. Dimension 0 holds the four gates in the
        # order `_step` unpacks them: input, forget, candidate, output.
        bound = 1.0 / math.sqrt(hidden_size)
        self.input_weights = nn.Parameter(
            torch.empty(4, hidden_size, input_size).uniform_(-bound, bound)
        )
        self.hidden_weights = nn.Parameter(
            torch.empty(4, hidden_size, hidden_size).uniform_(-bound, bound)
        )
        self.bias = nn.Parameter(torch.empty(4, hidden_size).uniform_(-bound, bound))

    def sample_masks(self, rows: int) -> DropoutMasks:
        """Sample the dropout masks of every row, keeping each feature with probability
        `1 - dropout`.

        Args:
            rows: The number of rows, each with its own masks.

        Returns:
            DropoutMasks: The masks on the input and on the hidden state.
        """

        def mask(width: int) -> Float[torch.Tensor, 'B 4 W']:
            keep = torch.rand(size=(rows, 4, width), device=self.bias.device) >= self.dropout
            return keep.to(dtype=self.bias.dtype) / (1.0 - self.dropout)

        return DropoutMasks(input=mask(self.input_size), hidden=mask(self.hidden_size))

    def _step(
        self,
        projected_inputs: Float[torch.Tensor, 'B 4 H'],
        hidden: Float[torch.Tensor, 'B 4 H'],
        cell: Float[torch.Tensor, 'B H'],
    ) -> LSTMState:
        """One LSTM step from an input already multiplied by the input weights and a hidden
        state already masked for each gate.

        Args:
            projected_inputs: The input's contribution to each gate.
            hidden: The previous hidden state, as each gate reads it.
            cell: The previous cell state.

        Returns:
            LSTMState: The next hidden and cell states.
        """
        gates = (
            projected_inputs + torch.einsum('bgk,ghk->bgh', hidden, self.hidden_weights) + self.bias
        )
        input_gate, forget_gate, candidate, output_gate = gates.unbind(dim=1)  # 4 x [B, H]
        cell = forget_gate.sigmoid() * cell + input_gate.sigmoid() * candidate.tanh()
        return LSTMState(hidden=output_gate.sigmoid() * cell.tanh(), cell=cell)

    def forward(
        self,
        inputs: Float[torch.Tensor, 'B T I'],
        state: LSTMState,
        *,
        masks: DropoutMasks,
        active: Bool[torch.Tensor, 'B T'] | None = None,
    ) -> tuple[Float[torch.Tensor, 'B T H'], LSTMState]:
        """Run the layer over a sequence of `T` inputs per row, starting from `state`.

        Args:
            inputs: One input per position.
            state: The hidden and cell states before the first position.
            masks: The dropout masks, shared by every position.
            active: True where a row reads a real input and False where it keeps its state, such
                as at PAD. None when every row reads a real input at every position.

        Returns:
            The hidden state after each position, and the final state.
        """
        gate_inputs = inputs.unsqueeze(dim=2).expand(-1, -1, 4, -1)  # [B, T, I] -> [B, T, 4, I]
        gate_inputs = gate_inputs * masks.input.unsqueeze(dim=1)
        projected = torch.einsum('btgi,ghi->btgh', gate_inputs, self.input_weights)
        hidden, cell = state
        outputs = []
        for position in range(inputs.size(dim=1)):
            gate_hidden = hidden.unsqueeze(dim=1).expand(-1, 4, -1) * masks.hidden  # [B, 4, H]
            next_hidden, next_cell = self._step(
                projected_inputs=projected[:, position], hidden=gate_hidden, cell=cell
            )
            if active is not None:
                reading = active[:, position].unsqueeze(dim=1)  # [B, 1]
                next_hidden = torch.where(condition=reading, input=next_hidden, other=hidden)
                next_cell = torch.where(condition=reading, input=next_cell, other=cell)
            hidden, cell = next_hidden, next_cell
            outputs.append(hidden)
        # T x [B, H] -> [B, T, H]
        return torch.stack(outputs, dim=1), LSTMState(hidden=hidden, cell=cell)


class VariationalLSTM(nn.Module):
    """Stacked `VariationalLSTMLayer` layers, each reading the outputs of the layer below."""

    def __init__(self, *, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        """
        Args:
            input_size: Width of the bottom layer's inputs, `I`.
            hidden_size: Width of every layer's states, `H`.
            num_layers: Number of stacked layers.
            dropout: Probability of dropping a gate input or hidden feature.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList(
            VariationalLSTMLayer(
                input_size=input_size if layer == 0 else hidden_size,
                hidden_size=hidden_size,
                dropout=dropout,
            )
            for layer in range(num_layers)
        )

    def sample_masks(self, rows: int) -> list[DropoutMasks]:
        """Sample the dropout masks of every layer, bottom first.

        Args:
            rows: The number of rows, each with its own masks.

        Returns:
            list[DropoutMasks]: The masks of every layer.
        """
        return [layer.sample_masks(rows) for layer in self.layers]

    def forward_from_state(
        self,
        inputs: Float[torch.Tensor, 'B T I'],
        state: LSTMState,
        *,
        masks: list[DropoutMasks],
        active: Bool[torch.Tensor, 'B T'] | None = None,
    ) -> tuple[Float[torch.Tensor, 'B T H'], list[LSTMState]]:
        """Run the stack from a single state, such as the encoder's summary of a prefix: the
        bottom layer starts from `state`, and every upper layer from the final state of the
        layer below.

        Args:
            inputs: One input per position.
            state: The bottom layer's state before the first position.
            masks: The dropout masks of every layer.
            active: True where a row reads a real input and False where it keeps its state, such
                as at PAD. None when every row reads a real input at every position.

        Returns:
            The top layer's hidden state after each position, and every layer's final state.
        """
        states = []
        for layer, layer_masks in zip(self.layers, masks, strict=True):
            inputs, state = layer(inputs=inputs, state=state, masks=layer_masks, active=active)
            states.append(state)
        return inputs, states

    def forward(
        self,
        inputs: Float[torch.Tensor, 'B T I'],
        states: list[LSTMState],
        *,
        masks: list[DropoutMasks],
    ) -> tuple[Float[torch.Tensor, 'B T H'], list[LSTMState]]:
        """Continue the stack where it left off, every layer from its own state.

        Args:
            inputs: One input per position.
            states: Every layer's state before the first position, bottom first.
            masks: The dropout masks of every layer.

        Returns:
            The top layer's hidden state after each position, and every layer's final state.
        """
        new_states = []
        for layer, state, layer_masks in zip(self.layers, states, masks, strict=True):
            inputs, state = layer(inputs=inputs, state=state, masks=layer_masks)
            new_states.append(state)
        return inputs, new_states
