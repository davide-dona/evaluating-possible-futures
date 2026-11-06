from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.datasets.dataset import EncodedLog, test_prefix_batches
from src.datasets.vocabulary import PAD, Vocabulary
from src.models.u_ed_lstm import UEDLSTM

CASE_COLUMN = 'case'
PREFIX_LENGTH_COLUMN = 'prefix_length'
DRAW_COLUMN = 'draw'
ACTIVITIES_COLUMN = 'activities'
TRUNCATED_COLUMN = 'truncated'

SCHEMA = pa.schema(
    [
        (CASE_COLUMN, pa.string()),
        (PREFIX_LENGTH_COLUMN, pa.int32()),
        (DRAW_COLUMN, pa.int32()),
        (ACTIVITIES_COLUMN, pa.list_(pa.dictionary(pa.int16(), pa.string()))),
        (TRUNCATED_COLUMN, pa.bool_()),
    ]
)


def generate(
    model: UEDLSTM,
    log: EncodedLog,
    vocabulary: Vocabulary,
    *,
    path: Path,
    sampled: bool,
    num_samples: int,
    max_rows: int,
    device: torch.device,
) -> int:
    """Write the suffixes drawn for every scored test prefix, one row per draw.

    The draws of a prefix are consecutive rows. A draw holds at most `max_case_length` events.

    Args:
        model: The trained model, on `device`.
        log: The test split.
        vocabulary: The vocabulary the model was trained with.
        path: The Parquet file to write.
        sampled: Whether to draw `num_samples` suffixes by `UEDLSTM.sample`, or one by
            `UEDLSTM.greedy`.
        num_samples: Draws per prefix when sampled.
        max_rows: The most draws decoded at once.
        device: Where the model runs.

    Returns:
        The number of prefixes written.
    """
    draws = num_samples if sampled else 1
    names = pa.array(vocabulary.names, type=pa.string())
    path.parent.mkdir(parents=True, exist_ok=True)
    prefixes_written = 0
    with pq.ParquetWriter(path, schema=SCHEMA) as writer:
        for cuts, prefixes in test_prefix_batches(log, batch_size=max(1, max_rows // draws)):
            prefixes = prefixes.to(device)
            max_steps = vocabulary.max_case_length
            if sampled:
                activities, truncated = model.sample(
                    prefixes, num_samples=num_samples, max_steps=max_steps
                )
            else:
                activities, truncated = model.greedy(prefixes, max_steps=max_steps)
            activities = activities.flatten(end_dim=1).cpu().numpy()  # [B, S, L] -> [B * S, L]
            real = activities != PAD
            offsets = np.concatenate([[0], np.cumsum(real.sum(axis=1))]).astype(np.int32)
            values = pa.DictionaryArray.from_arrays(
                indices=pa.array(activities[real].astype(np.int16)), dictionary=names
            )
            columns = {
                CASE_COLUMN: pa.array(np.repeat(log.cases[cuts.case], draws), type=pa.string()),
                PREFIX_LENGTH_COLUMN: pa.array(np.repeat(cuts.length, draws).astype(np.int32)),
                DRAW_COLUMN: pa.array(np.tile(np.arange(draws, dtype=np.int32), len(cuts.case))),
                ACTIVITIES_COLUMN: pa.ListArray.from_arrays(
                    offsets=pa.array(offsets), values=values
                ),
                TRUNCATED_COLUMN: pa.array(truncated.flatten().cpu().numpy()),
            }
            writer.write_table(pa.table(columns, schema=SCHEMA))
            prefixes_written += len(cuts.case)
    return prefixes_written
