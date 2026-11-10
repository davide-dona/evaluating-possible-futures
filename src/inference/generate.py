from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from src.datasets.dataset import Cuts, EncodedLog, test_prefix_batches
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

# The first character an activity is encoded as, in the Unicode private use area.
_FIRST_CHARACTER = 0xE000


class Generations(NamedTuple):
    """The draws of a file in `SCHEMA`, each also as a string of one character per activity,
    so that equal suffixes are equal strings."""

    table: pa.Table  # the rows of the file
    suffixes: list[str]  # the drawn suffix of every row
    characters: dict[str, str]  # the character of every activity name

    def prefixes(self) -> Iterator[tuple[int, int]]:
        """The rows of every prefix, as its first row and the row past its last."""
        first_draws = pc.equal(self.table[DRAW_COLUMN], 0).to_numpy(zero_copy_only=False)
        starts = np.flatnonzero(first_draws)
        return zip(starts, np.append(starts[1:], len(self.table)), strict=True)


def read_generations(path: Path, *, other_names: Iterable[str] = ()) -> Generations:
    """Read a file written in `SCHEMA`.

    Args:
        path: The Parquet file.
        other_names: Activity names to give a character besides those of the file, such as the
            activities of the observed suffixes the draws are compared with.
    """
    table = pq.read_table(path).unify_dictionaries().combine_chunks()
    activities = table[ACTIVITIES_COLUMN].chunk(0)
    names = activities.values.dictionary.to_pylist()
    unique = dict.fromkeys([*names, *other_names])
    characters = {name: chr(_FIRST_CHARACTER + i) for i, name in enumerate(unique)}
    of_index = np.array([characters[name] for name in names], dtype=object)
    drawn = of_index[activities.values.indices.to_numpy(zero_copy_only=False)]
    offsets = activities.offsets.to_numpy()
    suffixes = [''.join(drawn[a:b]) for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
    return Generations(table=table, suffixes=suffixes, characters=characters)


def generate(
    model: UEDLSTM,
    log: EncodedLog,
    vocabulary: Vocabulary,
    *,
    path: Path,
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
        num_samples: Draws per prefix, by `UEDLSTM.sample`.
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
            activities, truncated = model.sample(
                prefixes, num_samples=num_samples, max_steps=vocabulary.max_case_length
            )
            table = suffix_table(
                log,
                cuts,
                activities=activities.flatten(end_dim=1).cpu().numpy(),  # [B, S, L] -> [B * S, L]
                truncated=truncated.flatten().cpu().numpy(),
                draws=num_samples,
                names=names,
            )
            writer.write_table(table)
            prefixes_written += len(cuts.case)
    return prefixes_written


def suffix_table(
    log: EncodedLog,
    cuts: Cuts,
    *,
    activities: np.ndarray,
    truncated: np.ndarray,
    draws: int,
    names: pa.Array,
) -> pa.Table:
    """The rows of `SCHEMA` for the draws of a batch of prefixes.

    Args:
        log: The split the prefixes are cut from.
        cuts: The prefixes.
        activities: The activity indices of every draw, PAD past its end, `[N * draws, L]`, the
            draws of a prefix consecutive.
        truncated: Whether each draw reached the step limit without ending, `[N * draws]`.
        draws: Draws per prefix.
        names: The name of every vocabulary index.
    """
    real = activities != PAD
    offsets = np.concatenate([[0], np.cumsum(real.sum(axis=1))]).astype(np.int32)
    values = pa.DictionaryArray.from_arrays(
        indices=pa.array(activities[real].astype(np.int16)), dictionary=names
    )
    columns = {
        CASE_COLUMN: pa.array(np.repeat(log.cases[cuts.case], draws), type=pa.string()),
        PREFIX_LENGTH_COLUMN: pa.array(np.repeat(cuts.length, draws).astype(np.int32)),
        DRAW_COLUMN: pa.array(np.tile(np.arange(draws, dtype=np.int32), len(cuts.case))),
        ACTIVITIES_COLUMN: pa.ListArray.from_arrays(offsets=pa.array(offsets), values=values),
        TRUNCATED_COLUMN: pa.array(truncated),
    }
    return pa.table(columns, schema=SCHEMA)
