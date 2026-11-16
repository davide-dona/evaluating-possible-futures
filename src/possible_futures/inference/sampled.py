from pathlib import Path

import torch

from possible_futures.datasets.encoded_log import EncodedLog
from possible_futures.datasets.vocabulary import Vocabulary
from possible_futures.inference.draws import DecodingWriter
from possible_futures.models.u_ed_lstm import UEDLSTM


def write_sampled_draws(
    model: UEDLSTM,
    test_log: EncodedLog,
    vocabulary: Vocabulary,
    *,
    path: Path,
    num_draws: int,
    max_steps: int,
    max_rows: int,
    device: torch.device,
) -> int:
    """Sample suffixes for every test prefix from the model and save them to `path`.

    Args:
        model: The trained model, on `device`.
        test_log: The test split.
        vocabulary: The vocabulary the model was trained with.
        path: The Parquet file to write.
        num_draws: Draws per prefix.
        max_steps: The most events a draw holds.
        max_rows: The most draws decoded at once.
        device: Where the model runs.

    Returns:
        The number of prefixes written.
    """
    num_prefixes = 0
    with DecodingWriter(path, vocabulary) as writer:
        for cuts in test_log.cut_batches_by_length(batch_size=max(1, max_rows // num_draws)):
            prefixes = test_log.prefixes(cuts).to(device)
            activities = model.sample(prefixes, num_draws=num_draws, max_steps=max_steps)
            writer.write(
                case_ids=test_log.case_ids[cuts.case_indices],
                prefix_lengths=cuts.prefix_lengths,
                activities=activities.flatten(end_dim=1).cpu().numpy(),  # [B, S, L] -> [B * S, L]
                num_draws=num_draws,
            )
            num_prefixes += len(cuts.case_indices)
    return num_prefixes
