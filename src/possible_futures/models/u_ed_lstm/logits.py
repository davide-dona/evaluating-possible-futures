from collections.abc import Sequence
from typing import NamedTuple, Self

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int64
from torch import nn

from possible_futures.datasets.tokenizer import SpecialToken


def to_classes(tokens: Int64[torch.Tensor, '*shape']) -> Int64[torch.Tensor, '*shape']:
    """The output class of every token, UNK and up.

    Args:
        tokens: The tokens, UNK and up.

    Returns:
        torch.Tensor: The output class of every token.
    """
    return tokens - SpecialToken.UNK


def to_tokens(classes: Int64[torch.Tensor, '*shape']) -> Int64[torch.Tensor, '*shape']:
    """The token of every output class.

    Args:
        classes: The output classes.

    Returns:
        torch.Tensor: The token of every class.
    """
    return classes + SpecialToken.UNK


class GaussianLogits(NamedTuple):
    """Independent Gaussian logits over the output classes, the tokens from UNK on."""

    mean: Float[torch.Tensor, '*batch C']
    log_variance: Float[torch.Tensor, '*batch C']

    @classmethod
    def stack(cls, steps: Sequence['GaussianLogits']) -> Self:
        """The logits of every step, stacked along a new dimension after the batch.

        Args:
            steps: The logits `[B, C]` of every step.

        Returns:
            GaussianLogits: The logits `[B, T, C]`.
        """
        return cls(
            mean=torch.stack([step.mean for step in steps], dim=1),  # T x [B, C] -> [B, T, C]
            log_variance=torch.stack([step.log_variance for step in steps], dim=1),
        )

    def draw(self, *, samples: int) -> Float[torch.Tensor, 'samples *batch C']:
        """Draw `samples` logit vectors per distribution."""
        noise = torch.randn(size=(samples, *self.mean.shape), device=self.mean.device)
        return self.mean + (0.5 * self.log_variance).exp() * noise

    def most_likely_tokens(self) -> Int64[torch.Tensor, '*batch']:
        """The token of the class with the highest mean logit."""
        return to_tokens(self.mean.argmax(dim=-1))

    def sample_tokens(self) -> Int64[torch.Tensor, ' B']:
        """Draw one logit vector per row of logits `[B, C]`, then a token from its softmax."""
        drawn = self.draw(samples=1).squeeze(dim=0)  # [1, B, C] -> [B, C]
        classes = torch.multinomial(input=drawn.softmax(dim=-1), num_samples=1)
        return to_tokens(classes.squeeze(dim=1))  # [B, 1] -> [B]

    def cross_entropy(
        self, targets: Int64[torch.Tensor, 'B T'], *, samples: int
    ) -> Float[torch.Tensor, '']:
        """The cross-entropy of the targets under logits `[B, T, C]`, averaged over `samples`
        logit draws, the positions, and the batch.

        Args:
            targets: The token at every position.
            samples: The number of logit draws.

        Returns:
            torch.Tensor: The mean cross-entropy.
        """
        drawn = self.draw(samples=samples)  # [samples, B, T, C]
        classes = to_classes(targets).expand(samples, -1, -1)  # [B, T] -> [samples, B, T]
        return F.cross_entropy(input=drawn.flatten(end_dim=2), target=classes.flatten())


class GaussianHead(nn.Module):
    """Maps hidden states to Gaussian logits over the output classes."""

    def __init__(self, *, hidden_size: int, num_classes: int) -> None:
        """
        Args:
            hidden_size: Width of every hidden state.
            num_classes: Number of output classes.
        """
        super().__init__()
        self.mean = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.log_variance = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, features: Float[torch.Tensor, 'B H']) -> GaussianLogits:
        """The logits `[B, C]` of every row of `features`."""
        return GaussianLogits(mean=self.mean(features), log_variance=self.log_variance(features))
