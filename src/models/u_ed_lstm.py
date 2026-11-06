from pathlib import Path
from typing import Any, Self

import torch
import torch.nn.functional as F
from torch import nn

from src.datasets.dataset import Batch, Prefixes
from src.datasets.vocabulary import EOS, PAD
from src.models.lstm import GateDropoutLSTMCell, GateMasks, LSTMState


def embedding_dim(num_activities: int) -> int:
    """The width of the activity embeddings, by the rule of thumb of the original U-ED-LSTM."""
    return min(600, round(1.6 * (num_activities + 2) ** 0.56))


class UEDLSTM(nn.Module):
    """Activity-only U-ED-LSTM: an encoder-decoder LSTM with Gaussian activity logits and
    variational dropout kept active in sampling.

    The encoder layers are chained: each reads the outputs of the layer below and starts from its
    final state. The decoder's first input is the last prefix event; at its first step the bottom
    layer starts from the encoder's final state and each upper layer from the state the layer
    below just reached, and from then on every layer carries its own state.
    """

    def __init__(
        self,
        *,
        vocabulary_size: int,
        num_activities: int,
        num_layers: int,
        hidden_size: int,
        dropout: float,
        logit_samples: int,
    ) -> None:
        """
        Args:
            vocabulary_size: Number of input indices, special ones included.
            num_activities: Number of activities, which with EOS are the output classes.
            num_layers: Layers of the encoder and of the decoder.
            hidden_size: Width of every LSTM state.
            dropout: Probability of dropping a gate input or hidden feature.
            logit_samples: Gaussian logit draws the loss averages over.
        """
        super().__init__()
        self.config = {
            'vocabulary_size': vocabulary_size,
            'num_activities': num_activities,
            'num_layers': num_layers,
            'hidden_size': hidden_size,
            'dropout': dropout,
            'logit_samples': logit_samples,
        }
        self.dropout = dropout
        self.logit_samples = logit_samples
        width = embedding_dim(num_activities)
        self.encoder_embedding = nn.Embedding(
            num_embeddings=vocabulary_size, embedding_dim=width, padding_idx=PAD
        )
        self.decoder_embedding = nn.Embedding(
            num_embeddings=vocabulary_size, embedding_dim=width, padding_idx=PAD
        )
        self.encoder_cells = _stack(input_size=width, hidden_size=hidden_size, layers=num_layers)
        self.decoder_cells = _stack(input_size=width, hidden_size=hidden_size, layers=num_layers)
        self.mean_head = nn.Linear(in_features=hidden_size, out_features=num_activities + 1)
        self.log_variance_head = nn.Linear(in_features=hidden_size, out_features=num_activities + 1)

    def encode(self, prefixes: Prefixes, *, masks: list[GateMasks] | None) -> LSTMState:
        """Read the real events of prefixes `[B, P]` and return the top layer's final state."""
        inputs = self.encoder_embedding(prefixes.activities)  # [B, P] -> [B, P, D]
        positions = torch.arange(end=inputs.size(dim=1), device=inputs.device)
        keep = positions.unsqueeze(dim=0) < prefixes.length.unsqueeze(dim=1)  # [B, P]
        zeros = inputs.new_zeros(size=(inputs.size(dim=0), self.config['hidden_size']))
        state = (zeros, zeros)
        for layer, cell in enumerate(self.encoder_cells):
            inputs, state = cell(
                inputs, state, masks=None if masks is None else masks[layer], keep=keep
            )
        return state

    def decode_step(
        self,
        activities: torch.Tensor,
        states: LSTMState | list[LSTMState],
        *,
        masks: list[GateMasks] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[LSTMState]]:
        """Read one decoder input per row and predict the next event's logit distribution.

        Args:
            activities: `[B]` the previous event of every row.
            states: The encoder's final state at the first step, then each layer's own state.
            masks: The decoder's dropout masks, one per layer, or None for no dropout.

        Returns:
            The logit means and log-variances `[B, C]` over EOS and every activity, and each
            layer's new state.
        """
        inputs = self.decoder_embedding(activities).unsqueeze(dim=1)  # [B] -> [B, 1, D]
        first_step = isinstance(states, tuple)
        below = states
        new_states = []
        for layer, cell in enumerate(self.decoder_cells):
            state = below if first_step else states[layer]
            inputs, state = cell(inputs, state, masks=None if masks is None else masks[layer])
            new_states.append(state)
            below = state
        features = inputs.squeeze(dim=1)  # [B, 1, H] -> [B, H]
        return self.mean_head(features), self.log_variance_head(features), new_states

    def forward(self, batch: Batch, *, teacher_forcing: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the logit distribution of each target position, dropout active.

        Each step after the first reads the previous target with probability `teacher_forcing`,
        one coin for the whole batch, and otherwise the most likely class of the previous step.

        Returns:
            The logit means and log-variances, `[B, W, C]` each.
        """
        prefixes, targets = batch
        rows = targets.size(dim=0)
        encoded = self.encode(prefixes, masks=self._masks(self.encoder_cells, rows=rows))
        decoder_masks = self._masks(self.decoder_cells, rows=rows)
        inputs = _last_events(prefixes)
        states: LSTMState | list[LSTMState] = encoded
        means, log_variances = [], []
        for position in range(targets.size(dim=1)):
            if position > 0:
                if torch.rand(size=()).item() < teacher_forcing:
                    inputs = targets[:, position - 1]
                else:
                    inputs = means[-1].argmax(dim=-1) + EOS
            mean, log_variance, states = self.decode_step(inputs, states, masks=decoder_masks)
            means.append(mean)
            log_variances.append(log_variance)
        return torch.stack(means, dim=1), torch.stack(log_variances, dim=1)  # W x [B, C]

    def loss(
        self, means: torch.Tensor, log_variances: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """The cross-entropy of the targets `[B, W]`, averaged over `logit_samples` Gaussian
        logit draws, the positions, and the batch."""
        noise = torch.randn(size=(self.logit_samples, *means.shape), device=means.device)
        logits = means + (0.5 * log_variances).exp() * noise  # [S, B, W, C]
        classes = (targets - EOS).expand(self.logit_samples, -1, -1)  # [B, W] -> [S, B, W]
        return F.cross_entropy(input=logits.flatten(end_dim=2), target=classes.flatten())

    def weight_penalty(self) -> torch.Tensor:
        """The sum of the squares of every parameter."""
        return sum(parameter.square().sum() for parameter in self.parameters())

    @torch.no_grad()
    def sample(
        self, prefixes: Prefixes, *, num_samples: int, max_steps: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw `num_samples` suffixes per prefix.

        Each draw encodes its prefix under its own dropout masks, and at every step draws new
        decoder masks, Gaussian logits around the means, and an activity from their softmax.

        Returns:
            The activity indices `[B, S, L]`, PAD past each draw's end, and `[B, S]` whether a
            draw reached `max_steps` without EOS.
        """
        return self._generate(prefixes, num_samples=num_samples, max_steps=max_steps, sampled=True)

    @torch.no_grad()
    def greedy(self, prefixes: Prefixes, *, max_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode one suffix per prefix with dropout off, taking the class of the highest logit
        mean at every step.

        Returns:
            As `sample`, with `S = 1`.
        """
        return self._generate(prefixes, num_samples=1, max_steps=max_steps, sampled=False)

    def _generate(
        self, prefixes: Prefixes, *, num_samples: int, max_steps: int, sampled: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = prefixes.length.size(dim=0)
        rows, device = batch_size * num_samples, prefixes.length.device
        prefixes = Prefixes(*(field.repeat_interleave(num_samples, dim=0) for field in prefixes))
        encoded = self.encode(
            prefixes, masks=self._masks(self.encoder_cells, rows=rows) if sampled else None
        )
        states: LSTMState | list[LSTMState] = encoded
        inputs = _last_events(prefixes)
        activities = torch.full(size=(rows, max_steps), fill_value=PAD, device=device)
        # The output row of every row still decoding.
        live = torch.arange(end=rows, device=device)
        for position in range(max_steps):
            masks = self._masks(self.decoder_cells, rows=len(live)) if sampled else None
            mean, log_variance, states = self.decode_step(inputs, states, masks=masks)
            if sampled:
                logits = mean + (0.5 * log_variance).exp() * torch.randn_like(mean)
                classes = torch.multinomial(input=logits.softmax(dim=-1), num_samples=1)
                classes = classes.squeeze(dim=1)  # [B, 1] -> [B]
            else:
                classes = mean.argmax(dim=-1)
            emitted = classes + EOS
            continuing = (emitted != EOS).nonzero().squeeze(dim=1)
            live, inputs = live[continuing], emitted[continuing]
            activities[live, position] = inputs
            states = [(hidden[continuing], cell[continuing]) for hidden, cell in states]
            if len(live) == 0:
                break
        truncated = torch.zeros(size=(rows,), dtype=torch.bool, device=device)
        truncated[live] = True
        width = int((activities != PAD).sum(dim=1).max())
        return (
            activities[:, :width].view(batch_size, num_samples, width),  # [B * S, L] -> [B, S, L]
            truncated.view(batch_size, num_samples),
        )

    def _masks(self, cells: nn.ModuleList, *, rows: int) -> list[GateMasks]:
        device = self.mean_head.weight.device
        return [cell.sample_masks(rows=rows, p=self.dropout, device=device) for cell in cells]

    def save(self, path: Path) -> None:
        """Write the weights and the constructor arguments."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'config': self.config, 'state_dict': self.state_dict()}, path)

    @classmethod
    def load(cls, path: Path, *, device: torch.device) -> Self:
        """Rebuild a model written by `save`."""
        checkpoint: dict[str, Any] = torch.load(path, map_location=device)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model.to(device)


def _stack(*, input_size: int, hidden_size: int, layers: int) -> nn.ModuleList:
    return nn.ModuleList(
        GateDropoutLSTMCell(
            input_size=input_size if layer == 0 else hidden_size, hidden_size=hidden_size
        )
        for layer in range(layers)
    )


def _last_events(prefixes: Prefixes) -> torch.Tensor:
    """The last real event of every prefix, `[B, P] -> [B]`."""
    return prefixes.activities.gather(dim=1, index=(prefixes.length - 1).unsqueeze(dim=1)).squeeze(
        dim=1
    )
