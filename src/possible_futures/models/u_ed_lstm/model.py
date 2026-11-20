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


def _embedding_dim(vocabulary_size: int) -> int:
    """The embedding width for a vocabulary: grows sublinearly with it, up to 600.

    Args:
        vocabulary_size: The number of tokens, special tokens included, `V`.

    Returns:
        int: The width of every embedding, `D`.
    """
    return min(600, round(1.6 * vocabulary_size**0.56))


class Loss(NamedTuple):
    """The training objective of one batch and its data term.

    Attributes:
        total: The value to backpropagate, the data term plus the weight penalty.
        data: How poorly the model predicts the targets, the mean cross-entropy.
    """

    total: Float[torch.Tensor, '']
    data: float


class UEDLSTM(nn.Module):
    """U-ED-LSTM, the uncertainty-aware encoder-decoder LSTM, predicting activities only.

    The encoder summarizes a prefix; the decoder then writes the suffix one token at a time,
    starting from the last prefix token. The model expresses its uncertainty in two ways that
    stay active when sampling: dropout, which samples a slightly different network for every
    suffix, and Gaussian logits, which sample a different next-token distribution at every step.
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
            vocabulary_size: Number of tokens, special tokens included, `V`.
            num_layers: Layers of the encoder and of the decoder.
            hidden_size: Width of every LSTM state, `H`.
            dropout: Probability of dropping a gate input or hidden feature.
            logit_samples: Logit vectors sampled at every position to average the loss, `K`.
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
        width = _embedding_dim(vocabulary_size)

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
        head = GaussianHead(hidden_size=hidden_size, vocabulary_size=vocabulary_size)
        self.encoder = Encoder(embedding=encoder_embedding, lstm=encoder_lstm)
        self.decoder = Decoder(embedding=decoder_embedding, lstm=decoder_lstm, head=head)

    def forward(
        self, prefixes: PrefixTokens, targets: Int64[torch.Tensor, 'B T'], *, forced: Sequence[bool]
    ) -> GaussianLogits:
        """Predict the Gaussian logits of every target token, dropout active.

        Args:
            prefixes: The prefix tokens.
            targets: The tokens following each prefix.
            forced: Whether each step after the first is teacher forced, `T - 1` flags. Entry
                `t` decides what step `t + 1` reads: the true token `targets[:, t]` if True,
                the model's own most likely token at step `t` if False.

        Returns:
            GaussianLogits: The logits `[B, T, V]` of every target position.
        """
        rows = targets.size(dim=0)
        encoded = self.encoder(prefixes, masks=self.encoder.lstm.sample_masks(rows))
        masks = self.decoder.lstm.sample_masks(rows)
        logits, states = self.decoder.start(
            tokens=prefixes.last_tokens(), encoded=encoded, masks=masks
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
        """The training objective: the cross-entropy of the targets, averaged over
        `logit_samples` sampled logit vectors, the positions, and the batch, plus
        `weight_penalty` times the sum of the squared parameters.

        Args:
            logits: The logits `[B, T, V]` of every target position.
            targets: The tokens following each prefix.
            weight_penalty: The weight of the sum of the squared parameters.

        Returns:
            Loss: The total to backpropagate and its data term.
        """
        data = logits.cross_entropy(targets, num_samples=self.logit_samples)
        penalty = weight_penalty * sum(parameter.square().sum() for parameter in self.parameters())
        return Loss(total=data + penalty, data=data.item())

    @torch.no_grad()
    def sample_suffixes(
        self, prefixes: PrefixTokens, *, num_draws: int, max_steps: int
    ) -> Int64[torch.Tensor, 'B S L']:
        """Sample `num_draws` suffixes per prefix, each one token at a time until EOS.

        Every suffix encodes its prefix under its own dropout masks. At every step it samples
        new decoder masks, a logit vector from the Gaussian logits, and a token from that
        vector's softmax.

        Args:
            prefixes: The prefix tokens, `B` prefixes.
            num_draws: The number of suffixes sampled for each prefix, `S`.
            max_steps: The most tokens a suffix holds, EOS excluded.

        Returns:
            torch.Tensor: The tokens of every suffix, PAD past its end.
        """
        batch_size = prefixes.lengths.size(dim=0)
        rows, device = batch_size * num_draws, prefixes.lengths.device
        prefixes = prefixes.repeat_interleave(num_draws)
        encoded = self.encoder(prefixes, masks=self.encoder.lstm.sample_masks(rows))
        inputs = prefixes.last_tokens()
        tokens = torch.full(size=(rows, max_steps), fill_value=SpecialToken.PAD, device=device)
        # The row of `tokens` of every suffix still decoding.
        live = torch.arange(end=rows, device=device)
        states: list[LSTMState] = []
        for position in range(max_steps):
            masks = self.decoder.lstm.sample_masks(len(live))
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
