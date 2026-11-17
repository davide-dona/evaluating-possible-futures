from pathlib import Path

import torch

from possible_futures.datasets.encoded_log import EncodedLog
from possible_futures.datasets.vocabulary import Vocabulary
from possible_futures.inference.draws import DecodingWriter
from possible_futures.models.sampler import SuffixSampler


def write_draws(
    sampler: SuffixSampler,
    *,
    test_log: EncodedLog,
    vocabulary: Vocabulary,
    path: Path,
    num_draws: int,
    max_steps: int,
    max_rows: int,
    device: torch.device,
) -> int:
    """Draw suffixes for every test prefix from `sampler` and save them to `path`.

    Args:
        max_steps: The most events a draw holds.
        max_rows: The most draws held at once.
        device: Where the prefixes are sent to be sampled.

    Returns:
        The number of prefixes written.
    """
    num_prefixes = 0
    with DecodingWriter(path, vocabulary) as writer:
        for cuts in test_log.cut_batches_by_length(batch_size=max(1, max_rows // num_draws)):
            prefixes = test_log.prefixes(cuts).to(device)
            tokens = sampler.sample(prefixes, num_draws=num_draws, max_steps=max_steps)
            writer.write(
                case_ids=test_log.cases.case_ids[cuts.case_indices],
                prefix_lengths=cuts.prefix_lengths,
                tokens=tokens.flatten(end_dim=1).cpu().numpy(),  # [B, S, L] -> [B * S, L]
                num_draws=num_draws,
            )
            num_prefixes += len(cuts.case_indices)
    return num_prefixes
