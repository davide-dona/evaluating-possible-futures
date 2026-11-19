from collections.abc import Sequence
from typing import NamedTuple, Self

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int64
from torch import nn

from possible_futures.datasets.tokenizer import SpecialToken


class GaussianLogits(NamedTuple):
    """The model's uncertain prediction of the next token.

    Instead of one logit per token, the model predicts a Gaussian for every logit: a mean, its
    best guess, and a variance, how unsure it is. Each sample of these Gaussians is a plausible
    logit vector, whose softmax is a plausible next-token distribution. The `V` Gaussians of a
    position are independent.

    Attributes:
        mean: The mean of every logit.
        log_variance: The log of the variance of every logit.
    """

    mean: Float[torch.Tensor, '*batch V']
    log_variance: Float[torch.Tensor, '*batch V']

    @classmethod
    def stack(cls, steps: Sequence['GaussianLogits']) -> Self:
        """Join the predictions of consecutive decoding steps into one sequence.

        Args:
            steps: The logits `[B, V]` of every step, in order.

        Returns:
            GaussianLogits: The logits `[B, T, V]`, step `t` at position `t`.
        """
        return cls(
            mean=torch.stack([step.mean for step in steps], dim=1),  # T x [B, V] -> [B, T, V]
            log_variance=torch.stack([step.log_variance for step in steps], dim=1),
        )

    def sample_logits(self, *, num_samples: int) -> Float[torch.Tensor, 'K *batch V']:
        """Sample `num_samples` plausible logit vectors at every position, each the mean plus
        standard normal noise scaled by the standard deviation. Written this way, the samples
        let gradients reach both the mean and the variance.

        Args:
            num_samples: The number of logit vectors sampled at every position, `K`.

        Returns:
            torch.Tensor: The sampled logit vectors, sample after sample.
        """
        noise = torch.randn(size=(num_samples, *self.mean.shape), device=self.mean.device)
        return self.mean + (0.5 * self.log_variance).exp() * noise

    def most_likely_tokens(self) -> Int64[torch.Tensor, '*batch']:
        """The token the model favours at every position, ignoring its uncertainty.

        Returns:
            torch.Tensor: The token with the highest mean logit.
        """
        return self.mean.argmax(dim=-1)

    def sample_tokens(self) -> Int64[torch.Tensor, ' B']:
        """Sample one token per row in two stages: a plausible logit vector from the Gaussians,
        then a token from that vector's softmax. Both the model's uncertainty and the randomness
        of the process shape the result.

        Returns:
            torch.Tensor: One token per row of the logits `[B, V]`.
        """
        logits = self.sample_logits(num_samples=1).squeeze(dim=0)  # [1, B, V] -> [B, V]
        tokens = torch.multinomial(input=logits.softmax(dim=-1), num_samples=1)
        return tokens.squeeze(dim=1)  # [B, 1] -> [B]

    def cross_entropy(
        self, targets: Int64[torch.Tensor, 'B T'], *, num_samples: int
    ) -> Float[torch.Tensor, '']:
        """How poorly the logits `[B, T, V]` predict the targets: the cross-entropy, averaged
        over `num_samples` sampled logit vectors, the positions, and the batch.

        Args:
            targets: The token at every position, never PAD.
            num_samples: The number of logit vectors sampled at every position, `K`.

        Returns:
            torch.Tensor: The mean cross-entropy.
        """
        logits = self.sample_logits(num_samples=num_samples)  # [K, B, T, V]
        expanded = targets.expand(num_samples, -1, -1)  # [B, T] -> [K, B, T]
        return F.cross_entropy(input=logits.flatten(end_dim=2), target=expanded.flatten())


class GaussianHead(nn.Module):
    """Maps hidden states to Gaussian logits over the tokens. Tokens no model may predict, such
    as PAD, get a mean logit of minus infinity, so they are never predicted."""

    def __init__(self, *, hidden_size: int, vocabulary_size: int) -> None:
        """
        Args:
            hidden_size: Width of every hidden state, `H`.
            vocabulary_size: Number of tokens, special tokens included, `V`.
        """
        super().__init__()
        self.mean = nn.Linear(in_features=hidden_size, out_features=vocabulary_size)
        self.log_variance = nn.Linear(in_features=hidden_size, out_features=vocabulary_size)
        blocked = [token for token in SpecialToken if not token.is_output]
        self.register_buffer(
            name='blocked', tensor=torch.tensor(blocked, dtype=torch.int64), persistent=False
        )

    def forward(self, features: Float[torch.Tensor, 'B H']) -> GaussianLogits:
        """Predict the Gaussian logits of the next token from every hidden state.

        Args:
            features: The top LSTM layer's hidden state of every row.

        Returns:
            GaussianLogits: The logits `[B, V]` of every row.
        """
        mean = self.mean(features).index_fill(dim=-1, index=self.blocked, value=float('-inf'))
        return GaussianLogits(mean=mean, log_variance=self.log_variance(features))
