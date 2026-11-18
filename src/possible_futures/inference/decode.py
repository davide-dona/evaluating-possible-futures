from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from possible_futures.datasets.batches import PrefixBatch
from possible_futures.datasets.prefix_dataset import PrefixDataset
from possible_futures.datasets.vocabulary import Vocabulary
from possible_futures.inference.draws import DecodingWriter
from possible_futures.models.decoder import Decoder


def write_draws(
    decoder: Decoder,
    *,
    test: PrefixDataset,
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
    # Shortest first, so prefixes of similar length share a batch and little of it is padding.
    batches = DataLoader(
        test,
        batch_size=max(1, max_rows // num_draws),
        sampler=test.prefix_lengths().argsort(stable=True).tolist(),
        collate_fn=PrefixBatch.collate,
    )
    with DecodingWriter(path, vocabulary) as writer:
        for batch in batches:
            tokens = decoder.draw(
                batch.prefixes.to(device),
                num_draws=num_draws,
                max_steps=max_steps,
            )
            writer.write(
                case_ids=np.array(batch.case_ids, dtype=object),
                prefix_lengths=batch.prefixes.lengths.numpy(),
                tokens=tokens.flatten(end_dim=1).cpu().numpy(),  # [B, S, L] -> [B * S, L]
                num_draws=num_draws,
            )
    return len(test)
