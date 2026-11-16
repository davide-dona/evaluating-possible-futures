from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from possible_futures.datasets.dataset import EncodedLog, test_prefix_batches
from possible_futures.datasets.vocabulary import Vocabulary
from possible_futures.inference.generations import SCHEMA, suffix_table
from possible_futures.models.u_ed_lstm import UEDLSTM


def generate(
    model: UEDLSTM,
    log: EncodedLog,
    vocabulary: Vocabulary,
    *,
    path: Path,
    num_samples: int,
    max_steps: int,
    max_rows: int,
    device: torch.device,
) -> int:
    """Write the suffixes the model draws for every scored test prefix, in the `SCHEMA` of
    `possible_futures.inference.generations`.

    Args:
        model: The trained model, on `device`.
        log: The test split.
        vocabulary: The vocabulary the model was trained with.
        path: The Parquet file to write.
        num_samples: Draws per prefix, by `UEDLSTM.sample`.
        max_steps: The most events a draw holds.
        max_rows: The most draws decoded at once.
        device: Where the model runs.

    Returns:
        The number of prefixes written.
    """
    names = pa.array(vocabulary.names, type=pa.string())
    path.parent.mkdir(parents=True, exist_ok=True)
    prefixes_written = 0
    with pq.ParquetWriter(path, schema=SCHEMA) as writer:
        for cuts, prefixes in test_prefix_batches(log, batch_size=max(1, max_rows // num_samples)):
            prefixes = prefixes.to(device)
            activities = model.sample(prefixes, num_samples=num_samples, max_steps=max_steps)
            table = suffix_table(
                log,
                cuts,
                activities=activities.flatten(end_dim=1).cpu().numpy(),  # [B, S, L] -> [B * S, L]
                draws=num_samples,
                names=names,
            )
            writer.write_table(table)
            prefixes_written += len(cuts.case)
    return prefixes_written
