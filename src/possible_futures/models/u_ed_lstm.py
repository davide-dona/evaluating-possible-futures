from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple, Self

import torch
import torch.nn.functional as F
from torch import nn

from possible_futures.datasets.dataset import Batch, Prefixes
from possible_futures.datasets.vocabulary import EOS, PAD, UNK
from possible_futures.models.lstm import GateDropoutStack, GateMasks, LSTMState


def embedding_dim(num_classes: int) -> int:
    """The width of the embeddings of `num_classes` classes."""
    return min(600, round(1.6 * num_classes**0.56))


def _classes(activities: torch.Tensor) -> torch.Tensor:
    """The output class of every activity index, UNK and up."""
    return activities - UNK


def _activities(classes: torch.Tensor) -> torch.Tensor:
    """The activity index of every output class."""
    return classes + UNK


class GaussianLogits(NamedTuple):
    """Independent Gaussian logits over the output classes, `[..., C]` each."""

    mean: torch.Tensor
    log_variance: torch.Tensor

    def draw(self, *, samples: int) -> torch.Tensor:
        """Draw `samples` logit vectors per distribution, `[..., C] -> [S, ..., C]`."""
        noise = torch.randn(size=(samples, *self.mean.shape), device=self.mean.device)
        return self.mean + (0.5 * self.log_variance).exp() * noise


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
        num_layers: int,
        hidden_size: int,
        dropout: float,
        logit_samples: int,
    ) -> None:
        """
        Args:
            vocabulary_size: Number of input indices, special ones included. The output classes
                are the indices from UNK on.
            num_layers: Layers of the encoder and of the decoder.
            hidden_size: Width of every LSTM state.
            dropout: Probability of dropping a gate input or hidden feature.
            logit_samples: Gaussian logit draws the loss averages over.
        """
        super().__init__()
        self.arguments = {
            'vocabulary_size': vocabulary_size,
            'num_layers': num_layers,
            'hidden_size': hidden_size,
            'dropout': dropout,
            'logit_samples': logit_samples,
        }
        self.logit_samples = logit_samples
        num_classes = vocabulary_size - UNK
        width = embedding_dim(num_classes)
        self.encoder_embedding = nn.Embedding(
            num_embeddings=vocabulary_size, embedding_dim=width, padding_idx=PAD
        )
        self.decoder_embedding = nn.Embedding(
            num_embeddings=vocabulary_size, embedding_dim=width, padding_idx=PAD
        )
        self.encoder = GateDropoutStack(
            input_size=width, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout
        )
        self.decoder = GateDropoutStack(
            input_size=width, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout
        )
        self.mean_head = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.log_variance_head = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def encode(self, prefixes: Prefixes, *, masks: list[GateMasks]) -> LSTMState:
        """Read the real events of prefixes `[B, P]` and return the top layer's final state."""
        inputs = self.encoder_embedding(prefixes.activities)  # [B, P] -> [B, P, D]
        positions = torch.arange(end=inputs.size(dim=1), device=inputs.device)
        keep = positions.unsqueeze(dim=0) < prefixes.length.unsqueeze(dim=1)  # [B, P]
        zeros = inputs.new_zeros(size=(inputs.size(dim=0), self.encoder.hidden_size))
        _, states = self.encoder.chained(
            inputs=inputs, state=(zeros, zeros), masks=masks, keep=keep
        )
        return states[-1]

    def decode_first(
        self, activities: torch.Tensor, encoded: LSTMState, *, masks: list[GateMasks]
    ) -> tuple[GaussianLogits, list[LSTMState]]:
        """Read the first decoder input of every row, starting from the encoder's final state.

        Args:
            activities: `[B]` the last prefix event of every row.
            encoded: The encoder's final state.
            masks: The decoder's dropout masks, one per layer.

        Returns:
            The logits `[B, C]` of the next event, and each layer's new state.
        """
        inputs = self.decoder_embedding(activities).unsqueeze(dim=1)  # [B] -> [B, 1, D]
        outputs, states = self.decoder.chained(inputs=inputs, state=encoded, masks=masks)
        return self._logits(outputs), states

    def decode_step(
        self, activities: torch.Tensor, states: list[LSTMState], *, masks: list[GateMasks]
    ) -> tuple[GaussianLogits, list[LSTMState]]:
        """Read one decoder input per row after the first, every layer from its own state.

        Args:
            activities: `[B]` the previous event of every row.
            states: Each decoder layer's state.
            masks: The decoder's dropout masks, one per layer.

        Returns:
            The logits `[B, C]` of the next event, and each layer's new state.
        """
        inputs = self.decoder_embedding(activities).unsqueeze(dim=1)  # [B] -> [B, 1, D]
        outputs, states = self.decoder(inputs=inputs, states=states, masks=masks)
        return self._logits(outputs), states

    def _logits(self, outputs: torch.Tensor) -> GaussianLogits:
        features = outputs.squeeze(dim=1)  # [B, 1, H] -> [B, H]
        return GaussianLogits(
            mean=self.mean_head(features), log_variance=self.log_variance_head(features)
        )

    def forward(self, batch: Batch, *, forced: Sequence[bool]) -> GaussianLogits:
        """Predict the logit distribution of each target position, dropout active.

        Args:
            batch: Prefixes and their `[B, W]` targets.
            forced: `W - 1` flags; entry `t` is whether step `t + 1` reads target `t` instead
                of the most likely class of step `t`.

        Returns:
            The logits `[B, W, C]` of every target position.
        """
        prefixes, targets = batch
        rows = targets.size(dim=0)
        encoded = self.encode(prefixes, masks=self.encoder.sample_masks(rows))
        masks = self.decoder.sample_masks(rows)
        logits, states = self.decode_first(
            activities=prefixes.last_events(), encoded=encoded, masks=masks
        )
        steps = [logits]
        for position, force in enumerate(forced):
            if force:
                inputs = targets[:, position]
            else:
                inputs = _activities(logits.mean.argmax(dim=-1))
            logits, states = self.decode_step(activities=inputs, states=states, masks=masks)
            steps.append(logits)
        return GaussianLogits(
            mean=torch.stack([step.mean for step in steps], dim=1),  # W x [B, C] -> [B, W, C]
            log_variance=torch.stack([step.log_variance for step in steps], dim=1),
        )

    def loss(self, logits: GaussianLogits, targets: torch.Tensor) -> torch.Tensor:
        """The cross-entropy of the targets `[B, W]` under logits `[B, W, C]`, averaged over
        `logit_samples` Gaussian logit draws, the positions, and the batch."""
        drawn = logits.draw(samples=self.logit_samples)  # [S, B, W, C]
        classes = _classes(targets).expand(self.logit_samples, -1, -1)  # [B, W] -> [S, B, W]
        return F.cross_entropy(input=drawn.flatten(end_dim=2), target=classes.flatten())

    @torch.no_grad()
    def sample(self, prefixes: Prefixes, *, num_samples: int, max_steps: int) -> torch.Tensor:
        """Draw `num_samples` suffixes per prefix.

        Each draw encodes its prefix under its own dropout masks, and at every step draws new
        decoder masks, Gaussian logits around the means, and an activity from their softmax.

        Returns:
            The activity indices `[B, S, L]`, PAD past each draw's end.
        """
        batch_size = prefixes.length.size(dim=0)
        rows, device = batch_size * num_samples, prefixes.length.device
        prefixes = prefixes.repeat_interleave(num_samples)
        encoded = self.encode(prefixes, masks=self.encoder.sample_masks(rows))
        inputs = prefixes.last_events()
        activities = torch.full(size=(rows, max_steps), fill_value=PAD, device=device)
        # The output row of every row still decoding.
        live = torch.arange(end=rows, device=device)
        states: list[LSTMState] = []
        for position in range(max_steps):
            masks = self.decoder.sample_masks(len(live))
            if position == 0:
                logits, states = self.decode_first(activities=inputs, encoded=encoded, masks=masks)
            else:
                logits, states = self.decode_step(activities=inputs, states=states, masks=masks)
            drawn = logits.draw(samples=1).squeeze(dim=0)  # [1, B, C] -> [B, C]
            classes = torch.multinomial(input=drawn.softmax(dim=-1), num_samples=1)
            emitted = _activities(classes.squeeze(dim=1))  # [B, 1] -> [B]
            continuing = (emitted != EOS).nonzero().squeeze(dim=1)
            live, inputs = live[continuing], emitted[continuing]
            activities[live, position] = inputs
            states = [(hidden[continuing], cell[continuing]) for hidden, cell in states]
            if len(live) == 0:
                break
        width = int((activities != PAD).sum(dim=1).max())
        return activities[:, :width].view(batch_size, num_samples, width)  # [B * S, L] -> [B, S, L]

    def save(self, path: Path) -> None:
        """Write the weights and the constructor arguments."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'arguments': self.arguments, 'state_dict': self.state_dict()}, path)

    @classmethod
    def load(cls, path: Path, *, device: torch.device) -> Self:
        """Rebuild a model written by `save`."""
        checkpoint: dict[str, Any] = torch.load(path, map_location=device)
        model = cls(**checkpoint['arguments'])
        model.load_state_dict(checkpoint['state_dict'])
        return model.to(device).eval()
