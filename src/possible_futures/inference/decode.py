from pathlib import Path

import numpy as np
import torch

from possible_futures.datasets.encoding import EncodedPrefixes
from possible_futures.datasets.vocabulary import Vocabulary
from possible_futures.inference.draws import DecodingWriter
from possible_futures.models.decoder import Decoder


def write_draws(
    decoder: Decoder,
    *,
    test: EncodedPrefixes,
    vocabulary: Vocabulary,
    path: Path,
    num_draws: int,
    max_steps: int,
    max_rows: int,
    device: torch.device,
) -> int:
    """Draw suffixes for every test prefix from `decoder` and save them to `path`.

    Args:
        max_steps: The most events a draw holds.
        max_rows: The most draws held at once.
        device: Where the prefixes are sent to be decoded.

    Returns:
        The number of prefixes written.
    """
    batch_size = max(1, max_rows // num_draws)
    order = np.argsort(test.prefixes.lengths, kind='stable')
    with DecodingWriter(path, vocabulary) as writer:
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            tokens = decoder.draw(
                test.inputs(index).to(device), num_draws=num_draws, max_steps=max_steps
            )
            writer.write(
                case_ids=test.prefixes.case_ids[index],
                prefix_lengths=test.prefixes.lengths[index],
                tokens=tokens.flatten(end_dim=1).cpu().numpy(),  # [B, S, L] -> [B * S, L]
                num_draws=num_draws,
            )
    return len(test)
