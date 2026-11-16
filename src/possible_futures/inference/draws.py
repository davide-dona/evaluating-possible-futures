from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import NamedTuple, Self

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from possible_futures.datasets.vocabulary import PAD, Vocabulary

CASE_ID_COLUMN = 'case'
PREFIX_LENGTH_COLUMN = 'prefix_length'
DRAW_COLUMN = 'draw'
ACTIVITIES_COLUMN = 'activities'

# One row per draw, the draws of a prefix on consecutive rows and numbered from 0. The
# activities are dictionary encoded over the names of the vocabulary.
DRAWS_SCHEMA = pa.schema(
    [
        (CASE_ID_COLUMN, pa.string()),
        (PREFIX_LENGTH_COLUMN, pa.int32()),
        (DRAW_COLUMN, pa.int32()),
        (ACTIVITIES_COLUMN, pa.list_(pa.dictionary(pa.int16(), pa.string()))),
    ]
)


class PrefixDraws(NamedTuple):
    """The draws of one prefix of a decoding."""

    case_id: str
    prefix_length: int
    draws: list[tuple[int, ...]]  # the vocabulary indices of every drawn suffix


@dataclass(frozen=True, slots=True)
class Decoding:
    """The draws of every prefix of a decoding, in file order."""

    prefixes: list[PrefixDraws]

    def save(self, path: Path, vocabulary: Vocabulary) -> None:
        """Write the draws in `DRAWS_SCHEMA`, creating the directory if missing.

        Args:
            path: The Parquet file to write.
            vocabulary: The vocabulary the draws are encoded with.
        """
        draws = [draw for prefix in self.prefixes for draw in prefix.draws]
        counts = [len(prefix.draws) for prefix in self.prefixes]
        offsets = np.cumsum([0, *(len(draw) for draw in draws)])
        table = _draws_table(
            case_ids=np.repeat([prefix.case_id for prefix in self.prefixes], counts),
            prefix_lengths=np.repeat([prefix.prefix_length for prefix in self.prefixes], counts),
            draw_numbers=np.concatenate([np.arange(count) for count in counts], dtype=np.int32),
            offsets=offsets,
            values=np.fromiter((index for draw in draws for index in draw), dtype=np.int64),
            vocabulary=vocabulary,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)

    @classmethod
    def load(cls, path: Path, vocabulary: Vocabulary) -> Self:
        """Read a file written in `DRAWS_SCHEMA`.

        Args:
            path: The Parquet file.
            vocabulary: The vocabulary the file was written with.
        """
        table = pq.read_table(path).unify_dictionaries().combine_chunks()
        activities = table[ACTIVITIES_COLUMN].chunk(0)
        of_entry = vocabulary.encode(activities.values.dictionary.to_pylist())
        drawn = of_entry[activities.values.indices.to_numpy(zero_copy_only=False)]
        offsets = activities.offsets.to_numpy()
        suffixes = [
            tuple(drawn[a:b].tolist()) for a, b in zip(offsets[:-1], offsets[1:], strict=True)
        ]
        first_draws = pc.equal(table[DRAW_COLUMN], 0).to_numpy(zero_copy_only=False)
        starts = np.flatnonzero(first_draws)
        stops = np.append(starts[1:], len(table))
        case_ids = table[CASE_ID_COLUMN].take(pa.array(starts)).to_pylist()
        prefix_lengths = table[PREFIX_LENGTH_COLUMN].take(pa.array(starts)).to_pylist()
        return cls(
            prefixes=[
                PrefixDraws(case_id=case_id, prefix_length=prefix_length, draws=suffixes[a:b])
                for case_id, prefix_length, a, b in zip(
                    case_ids, prefix_lengths, starts, stops, strict=True
                )
            ]
        )


class DecodingWriter:
    """Writes a decoding in `DRAWS_SCHEMA` batch by batch, so no batch outlives its write."""

    def __init__(self, path: Path, vocabulary: Vocabulary) -> None:
        """
        Args:
            path: The Parquet file to write.
            vocabulary: The vocabulary the draws are encoded with.
        """
        self.path = path
        self.vocabulary = vocabulary

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(self.path, schema=DRAWS_SCHEMA)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._writer.close()

    def write(
        self,
        *,
        case_ids: np.ndarray,
        prefix_lengths: np.ndarray,
        activities: np.ndarray,
        num_draws: int,
    ) -> None:
        """Append the draws of a batch of prefixes.

        Args:
            case_ids: The case id of every prefix, `[N]`.
            prefix_lengths: The length of every prefix, `[N]`.
            activities: The activity indices of every draw, PAD past its end, `[N * num_draws,
                L]`, the draws of a prefix consecutive.
            num_draws: Draws per prefix.
        """
        real = activities != PAD
        table = _draws_table(
            case_ids=np.repeat(case_ids, num_draws),
            prefix_lengths=np.repeat(prefix_lengths, num_draws),
            draw_numbers=np.tile(np.arange(num_draws, dtype=np.int32), len(case_ids)),
            offsets=np.concatenate([[0], np.cumsum(real.sum(axis=1))]),
            values=activities[real],
            vocabulary=self.vocabulary,
        )
        self._writer.write_table(table)


def _draws_table(
    *,
    case_ids: np.ndarray,
    prefix_lengths: np.ndarray,
    draw_numbers: np.ndarray,
    offsets: np.ndarray,
    values: np.ndarray,
    vocabulary: Vocabulary,
) -> pa.Table:
    """The rows of `DRAWS_SCHEMA`, one per draw.

    Args:
        case_ids: The case id of every draw's prefix, `[R]`.
        prefix_lengths: The length of every draw's prefix, `[R]`.
        draw_numbers: The number of every draw among the draws of its prefix, `[R]`.
        offsets: Where the activities of every draw start in `values`, and where the last one
            stops, `[R + 1]`.
        values: The activity indices of every draw, one after the other.
        vocabulary: The vocabulary the activity indices belong to.
    """
    activities = pa.DictionaryArray.from_arrays(
        indices=pa.array(values.astype(np.int16)),
        dictionary=pa.array(vocabulary.names, type=pa.string()),
    )
    columns = {
        CASE_ID_COLUMN: pa.array(case_ids, type=pa.string()),
        PREFIX_LENGTH_COLUMN: pa.array(prefix_lengths.astype(np.int32)),
        DRAW_COLUMN: pa.array(draw_numbers),
        ACTIVITIES_COLUMN: pa.ListArray.from_arrays(
            offsets=pa.array(offsets.astype(np.int32)), values=activities
        ),
    }
    return pa.table(columns, schema=DRAWS_SCHEMA)
