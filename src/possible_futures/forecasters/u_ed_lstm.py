from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch

from possible_futures.cases import Activities
from possible_futures.datasets.collation import PrefixTokens
from possible_futures.datasets.tokenizer import Tokenizer
from possible_futures.forecasters.forecaster import Forecaster
from possible_futures.models.u_ed_lstm.model import UEDLSTM

_TOKENIZER_FILE = 'tokenizer.json'


@dataclass(frozen=True, slots=True, kw_only=True)
class UEDLSTMForecaster(Forecaster):
    """Samples suffixes from a trained U-ED-LSTM, translating activities to its tokens and back.

    Attributes:
        model: The trained model.
        tokenizer: The tokenizer the model was trained with.
    """

    model: UEDLSTM
    tokenizer: Tokenizer

    def save(self, directory: Path) -> None:
        """Write the model and the tokenizer it was trained with.

        Args:
            directory: The directory written into. Created if it does not exist.
        """
        self.model.save(directory)
        self.tokenizer.save(directory / _TOKENIZER_FILE)

    @classmethod
    def load(cls, directory: Path, *, device: torch.device) -> Self:
        """Rebuild a forecaster written by `save`.

        Args:
            directory: The directory `save` wrote into.
            device: The device the model is loaded onto.

        Returns:
            UEDLSTMForecaster: The model in evaluation mode, on `device`, with its tokenizer.
        """
        return cls(
            model=UEDLSTM.load(directory, device=device),
            tokenizer=Tokenizer.load(directory / _TOKENIZER_FILE),
        )

    def draw(
        self, prefixes: Sequence[Activities], *, num_draws: int, max_steps: int
    ) -> list[list[Activities]]:
        device = next(self.model.parameters()).device
        tokens = PrefixTokens.pad([self.tokenizer.encode(prefix) for prefix in prefixes])
        sampled = self.model.sample_suffixes(
            tokens.to(device), num_draws=num_draws, max_steps=max_steps
        )
        suffixes = self.tokenizer.decode(sampled.flatten(end_dim=1))  # [B, S, L] -> [B * S, L]
        return [suffixes[start : start + num_draws] for start in range(0, len(suffixes), num_draws)]
