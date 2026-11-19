from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch

from possible_futures.datasets.collation import PrefixTokens
from possible_futures.datasets.tokenizer import SpecialToken, Tokenizer
from possible_futures.forecasters.forecaster import Forecaster
from possible_futures.models.u_ed_lstm.model import UEDLSTM

_TOKENIZER_FILE = 'tokenizer.json'


@dataclass(frozen=True, slots=True, kw_only=True)
class UEDLSTMForecaster(Forecaster):
    """Draws suffixes from a U-ED-LSTM, mapping activities to its tokens and back."""

    model: UEDLSTM
    tokenizer: Tokenizer  # the tokenizer the model was trained on

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
        self, prefixes: Sequence[tuple[str, ...]], *, num_draws: int, max_steps: int
    ) -> list[list[tuple[str, ...]]]:
        device = next(self.model.parameters()).device
        tokens = PrefixTokens.pad([torch.from_numpy(self.tokenizer.encode(p)) for p in prefixes])
        drawn = self.model.draw(tokens.to(device), num_draws=num_draws, max_steps=max_steps)
        rows = drawn.flatten(end_dim=1).cpu().numpy()  # [B, S, L] -> [B * S, L]
        lengths = (rows != SpecialToken.PAD).sum(axis=1)
        suffixes = [
            self.tokenizer.decode(row[:length]) for row, length in zip(rows, lengths, strict=True)
        ]
        return [suffixes[start : start + num_draws] for start in range(0, len(suffixes), num_draws)]
