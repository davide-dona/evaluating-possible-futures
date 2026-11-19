import json
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple, Self

import torch
from jaxtyping import Float, Int64
from safetensors.torch import load_file, save_file
from torch import nn

from possible_futures.datasets.collation import PrefixTokens
from possible_futures.datasets.tokenizer import SpecialToken
from possible_futures.models.lstm import LSTMState, VariationalLSTM
from possible_futures.models.u_ed_lstm.decoder import Decoder
from possible_futures.models.u_ed_lstm.encoder import Encoder
from possible_futures.models.u_ed_lstm.logits import GaussianHead, GaussianLogits

_WEIGHTS_FILE = 'model.safetensors'
_ARGUMENTS_FILE = 'arguments.json'


def _embedding_dim(num_classes: int) -> int:
    """The width of the embeddings of `num_classes` classes.

    Args:
        num_classes: The number of output classes.

    Returns:
        int: The width of every embedding.
    """
    return min(600, round(1.6 * num_classes**0.56))


class Loss(NamedTuple):
    """The training objective: the `total` to backpropagate and the terms it sums."""

    total: Float[torch.Tensor, '']
    data: float
    penalty: float


class UEDLSTM(nn.Module):
    """Activity-only U-ED-LSTM: an encoder-decoder LSTM with Gaussian activity logits and
    variational dropout kept active in sampling. The decoder's first input is the last prefix
    event.
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
        width = _embedding_dim(num_classes)

        def embedding() -> nn.Embedding:
            return nn.Embedding(
                num_embeddings=vocabulary_size, embedding_dim=width, padding_idx=SpecialToken.PAD
            )

        def lstm() -> VariationalLSTM:
            return VariationalLSTM(
                input_size=width, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout
            )

        # The order the parts are built in sets the random init of every parameter.
        encoder_embedding, decoder_embedding = embedding(), embedding()
        encoder_lstm, decoder_lstm = lstm(), lstm()
        head = GaussianHead(hidden_size=hidden_size, num_classes=num_classes)
        self.encoder = Encoder(embedding=encoder_embedding, lstm=encoder_lstm)
        self.decoder = Decoder(embedding=decoder_embedding, lstm=decoder_lstm, head=head)

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
        encoded = self.encoder(prefixes, masks=self.encoder.sample_masks(rows))
        masks = self.decoder.sample_masks(rows)
        logits, states = self.decoder.start(
            tokens=prefixes.last_events(), encoded=encoded, masks=masks
        )
        steps = [logits]
        for position, force in enumerate(forced):
            inputs = targets[:, position] if force else logits.most_likely_tokens()
            logits, states = self.decoder.step(tokens=inputs, states=states, masks=masks)
            steps.append(logits)
        return GaussianLogits.stack(steps)

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
        data = logits.cross_entropy(targets, samples=self.logit_samples)
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
        encoded = self.encoder(prefixes, masks=self.encoder.sample_masks(rows))
        inputs = prefixes.last_events()
        tokens = torch.full(size=(rows, max_steps), fill_value=SpecialToken.PAD, device=device)
        # The output row of every row still decoding.
        live = torch.arange(end=rows, device=device)
        states: list[LSTMState] = []
        for position in range(max_steps):
            masks = self.decoder.sample_masks(len(live))
            if position == 0:
                logits, states = self.decoder.start(tokens=inputs, encoded=encoded, masks=masks)
            else:
                logits, states = self.decoder.step(tokens=inputs, states=states, masks=masks)
            emitted = logits.sample_tokens()
            continuing = (emitted != SpecialToken.EOS).nonzero().squeeze(dim=1)
            live, inputs = live[continuing], emitted[continuing]
            tokens[live, position] = inputs
            states = [state.select(continuing) for state in states]
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
