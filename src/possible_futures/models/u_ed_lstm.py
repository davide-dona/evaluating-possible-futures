import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Self

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int64
from safetensors.torch import load_file, save_file
from torch import nn

from possible_futures.datasets.collation import PrefixTokens
from possible_futures.datasets.tokenizer import SpecialToken, Tokenizer
from possible_futures.models.decoder import Decoder
from possible_futures.models.lstm import GateDropoutStack, GateMasks, LSTMState

_WEIGHTS_FILE = 'model.safetensors'
_ARGUMENTS_FILE = 'arguments.json'


def embedding_dim(num_classes: int) -> int:
    """The width of the embeddings of `num_classes` classes.

    Args:
        num_classes: The number of output classes.

    Returns:
        int: The width of every embedding.
    """
    return min(600, round(1.6 * num_classes**0.56))


def _to_classes(tokens: Int64[torch.Tensor, '*shape']) -> Int64[torch.Tensor, '*shape']:
    """The output class of every token, UNK and up.

    Args:
        tokens: The tokens, UNK and up.

    Returns:
        torch.Tensor: The output class of every token.
    """
    return tokens - SpecialToken.UNK


def _to_tokens(classes: Int64[torch.Tensor, '*shape']) -> Int64[torch.Tensor, '*shape']:
    """The token of every output class.

    Args:
        classes: The output classes.

    Returns:
        torch.Tensor: The token of every class.
    """
    return classes + SpecialToken.UNK


class GaussianLogits(NamedTuple):
    """Independent Gaussian logits over the output classes."""

    mean: Float[torch.Tensor, '*batch C']
    log_variance: Float[torch.Tensor, '*batch C']

    def draw(self, *, samples: int) -> Float[torch.Tensor, 'samples *batch C']:
        """Draw `samples` logit vectors per distribution."""
        noise = torch.randn(size=(samples, *self.mean.shape), device=self.mean.device)
        return self.mean + (0.5 * self.log_variance).exp() * noise


class Loss(NamedTuple):
    """The training objective: the `total` to backpropagate and the terms it sums."""

    total: Float[torch.Tensor, '']
    data: float
    penalty: float


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
            vocabulary_size: Number of tokens, special tokens included. The output classes are the
                tokens from UNK on.
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
        num_classes = vocabulary_size - SpecialToken.UNK
        width = embedding_dim(num_classes)
        self.encoder_embedding = nn.Embedding(
            num_embeddings=vocabulary_size, embedding_dim=width, padding_idx=SpecialToken.PAD
        )
        self.decoder_embedding = nn.Embedding(
            num_embeddings=vocabulary_size, embedding_dim=width, padding_idx=SpecialToken.PAD
        )
        self.encoder = GateDropoutStack(
            input_size=width, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout
        )
        self.decoder = GateDropoutStack(
            input_size=width, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout
        )
        self.mean_head = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.log_variance_head = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def encode(self, prefixes: PrefixTokens, *, masks: list[GateMasks]) -> LSTMState:
        """Read the real events of the prefixes.

        Args:
            prefixes: The prefix tokens.
            masks: The encoder's dropout masks, one per layer.

        Returns:
            LSTMState: The top layer's final state.
        """
        inputs = self.encoder_embedding(prefixes.tokens)  # [B, P] -> [B, P, D]
        positions = torch.arange(end=inputs.size(dim=1), device=inputs.device)
        advancing = positions.unsqueeze(dim=0) < prefixes.lengths.unsqueeze(dim=1)  # [B, P]
        zeros = inputs.new_zeros(size=(inputs.size(dim=0), self.encoder.hidden_size))
        _, states = self.encoder.forward_chained(
            inputs=inputs,
            state=LSTMState(hidden=zeros, cell=zeros),
            masks=masks,
            advancing=advancing,
        )
        return states[-1]

    def decode_first(
        self, tokens: Int64[torch.Tensor, ' B'], encoded: LSTMState, *, masks: list[GateMasks]
    ) -> tuple[GaussianLogits, list[LSTMState]]:
        """Read the first decoder input of every row, starting from the encoder's final state.

        Args:
            tokens: `[B]` the last prefix event of every row.
            encoded: The encoder's final state.
            masks: The decoder's dropout masks, one per layer.

        Returns:
            The logits `[B, C]` of the next event, and each layer's new state.
        """
        inputs = self.decoder_embedding(tokens).unsqueeze(dim=1)  # [B] -> [B, 1, D]
        outputs, states = self.decoder.forward_chained(inputs=inputs, state=encoded, masks=masks)
        return self._logits(outputs), states

    def decode_step(
        self, tokens: Int64[torch.Tensor, ' B'], states: list[LSTMState], *, masks: list[GateMasks]
    ) -> tuple[GaussianLogits, list[LSTMState]]:
        """Read one decoder input per row after the first, every layer from its own state.

        Args:
            tokens: `[B]` the previous event of every row.
            states: Each decoder layer's state.
            masks: The decoder's dropout masks, one per layer.

        Returns:
            The logits `[B, C]` of the next event, and each layer's new state.
        """
        inputs = self.decoder_embedding(tokens).unsqueeze(dim=1)  # [B] -> [B, 1, D]
        outputs, states = self.decoder(inputs=inputs, states=states, masks=masks)
        return self._logits(outputs), states

    def _logits(self, outputs: Float[torch.Tensor, 'B 1 H']) -> GaussianLogits:
        """The logit distribution of the next event, from the top decoder layer's outputs.

        Args:
            outputs: The top decoder layer's hidden state after the step.

        Returns:
            GaussianLogits: The logits `[B, C]` of the next event.
        """
        features = outputs.squeeze(dim=1)  # [B, 1, H] -> [B, H]
        return GaussianLogits(
            mean=self.mean_head(features), log_variance=self.log_variance_head(features)
        )

    def forward(
        self, prefixes: PrefixTokens, targets: Int64[torch.Tensor, 'B T'], *, forced: Sequence[bool]
    ) -> GaussianLogits:
        """Predict the logit distribution of each target position, dropout active.

        Args:
            prefixes: The prefix tokens.
            targets: The `[B, T]` events following each prefix.
            forced: `T - 1` flags; entry `t` is whether step `t + 1` reads target `t` instead
                of the most likely class of step `t`.

        Returns:
            The logits `[B, T, C]` of every target position.
        """
        rows = targets.size(dim=0)
        encoded = self.encode(prefixes, masks=self.encoder.sample_masks(rows))
        masks = self.decoder.sample_masks(rows)
        logits, states = self.decode_first(
            tokens=prefixes.last_events(), encoded=encoded, masks=masks
        )
        steps = [logits]
        for position, force in enumerate(forced):
            if force:
                inputs = targets[:, position]
            else:
                inputs = _to_tokens(logits.mean.argmax(dim=-1))
            logits, states = self.decode_step(tokens=inputs, states=states, masks=masks)
            steps.append(logits)
        return GaussianLogits(
            mean=torch.stack([step.mean for step in steps], dim=1),  # T x [B, C] -> [B, T, C]
            log_variance=torch.stack([step.log_variance for step in steps], dim=1),
        )

    def loss(
        self, logits: GaussianLogits, targets: Int64[torch.Tensor, 'B T'], *, weight_penalty: float
    ) -> Loss:
        """The full objective: the cross-entropy of the targets under the logits, averaged over
        `logit_samples` Gaussian logit draws, the positions, and the batch, plus `weight_penalty`
        times the sum of the squared parameters.

        Args:
            logits: The logits `[B, T, C]` of every target position.
            targets: The events following each prefix.
            weight_penalty: The weight of the sum of the squared parameters.

        Returns:
            Loss: The total to backpropagate and the terms it sums.
        """
        drawn = logits.draw(samples=self.logit_samples)  # [samples, B, T, C]
        # [B, T] -> [samples, B, T]
        classes = _to_classes(targets).expand(self.logit_samples, -1, -1)
        data = F.cross_entropy(input=drawn.flatten(end_dim=2), target=classes.flatten())
        penalty = weight_penalty * sum(parameter.square().sum() for parameter in self.parameters())
        return Loss(total=data + penalty, data=data.item(), penalty=penalty.item())

    @torch.no_grad()
    def draw(
        self, prefixes: PrefixTokens, *, num_draws: int, max_steps: int
    ) -> Int64[torch.Tensor, 'B S L']:
        """Draw `num_draws` suffixes per prefix.

        Each draw encodes its prefix under its own dropout masks, and at every step draws new
        decoder masks, Gaussian logits around the means, and a token from their softmax.

        Args:
            prefixes: The prefix tokens.
            num_draws: The number of suffixes drawn for each prefix.
            max_steps: The most events a suffix holds.

        Returns:
            torch.Tensor: The tokens `[B, S, L]` of every draw, PAD past its end.
        """
        batch_size = prefixes.lengths.size(dim=0)
        rows, device = batch_size * num_draws, prefixes.lengths.device
        prefixes = prefixes.repeat_interleave(num_draws)
        encoded = self.encode(prefixes, masks=self.encoder.sample_masks(rows))
        inputs = prefixes.last_events()
        tokens = torch.full(size=(rows, max_steps), fill_value=SpecialToken.PAD, device=device)
        # The output row of every row still decoding.
        live = torch.arange(end=rows, device=device)
        states: list[LSTMState] = []
        for position in range(max_steps):
            masks = self.decoder.sample_masks(len(live))
            if position == 0:
                logits, states = self.decode_first(tokens=inputs, encoded=encoded, masks=masks)
            else:
                logits, states = self.decode_step(tokens=inputs, states=states, masks=masks)
            drawn = logits.draw(samples=1).squeeze(dim=0)  # [1, B, C] -> [B, C]
            classes = torch.multinomial(input=drawn.softmax(dim=-1), num_samples=1)
            emitted = _to_tokens(classes.squeeze(dim=1))  # [B, 1] -> [B]
            continuing = (emitted != SpecialToken.EOS).nonzero().squeeze(dim=1)
            live, inputs = live[continuing], emitted[continuing]
            tokens[live, position] = inputs
            states = [
                LSTMState(hidden=hidden[continuing], cell=cell[continuing])
                for hidden, cell in states
            ]
            if len(live) == 0:
                break
        width = int((tokens != SpecialToken.PAD).sum(dim=1).max())
        # [B * S, L] -> [B, S, L]
        return tokens[:, :width].view(batch_size, num_draws, width)

    def save(self, directory: Path) -> None:
        """Write the weights and the constructor arguments.

        Args:
            directory: The directory written into. Created if it does not exist.
        """
        directory.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), directory / _WEIGHTS_FILE)
        (directory / _ARGUMENTS_FILE).write_text(json.dumps(self.arguments, indent=2) + '\n')

    @classmethod
    def load(cls, directory: Path, *, device: torch.device) -> Self:
        """Rebuild a model written by `save`.

        Args:
            directory: The directory `save` wrote into.
            device: The device the model is loaded onto.

        Returns:
            UEDLSTM: The model in evaluation mode, on `device`.
        """
        model = cls(**json.loads((directory / _ARGUMENTS_FILE).read_text()))
        model.load_state_dict(load_file(directory / _WEIGHTS_FILE, device=str(device)))
        return model.to(device).eval()


@dataclass(frozen=True, slots=True, kw_only=True)
class UEDLSTMDecoder(Decoder):
    """Draws suffixes from a U-ED-LSTM, mapping activities to its tokens and back."""

    model: UEDLSTM
    tokenizer: Tokenizer  # the tokenizer the model was trained on
    device: torch.device  # the device of the model

    def draw(
        self, prefixes: Sequence[tuple[str, ...]], *, num_draws: int, max_steps: int
    ) -> list[list[tuple[str, ...]]]:
        tokens = PrefixTokens.pad([torch.from_numpy(self.tokenizer.encode(p)) for p in prefixes])
        drawn = self.model.draw(tokens.to(self.device), num_draws=num_draws, max_steps=max_steps)
        rows = drawn.flatten(end_dim=1).cpu().numpy()  # [B, S, L] -> [B * S, L]
        lengths = (rows != SpecialToken.PAD).sum(axis=1)
        labels = np.array(self.tokenizer.labels, dtype=object)
        suffixes = [
            tuple(labels[row[:length]].tolist()) for row, length in zip(rows, lengths, strict=True)
        ]
        return [suffixes[start : start + num_draws] for start in range(0, len(suffixes), num_draws)]
