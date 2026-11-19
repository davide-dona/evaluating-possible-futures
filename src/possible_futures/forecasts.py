from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# One row per draw, the draws of a prefix on consecutive rows and numbered from 0.
FORECASTS_SCHEMA = pa.schema(
    [
        ('case_id', pa.string()),
        ('prefix_length', pa.int32()),
        ('draw', pa.int32()),
        ('activities', pa.list_(pa.dictionary(pa.int16(), pa.string()))),
    ]
)


@dataclass(frozen=True, slots=True)
class Forecast:
    """The suffixes drawn for one prefix of a case."""

    case_id: str
    prefix_length: int
    draws: list[tuple[str, ...]]  # the activities of every drawn suffix


def save_forecasts(forecasts: Sequence[Forecast], path: Path) -> None:
    """Write forecasts to a Parquet file.

    Args:
        forecasts: The forecasts written.
        path: The Parquet file written. Creates the parent directories if they do not exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_table(forecasts), path)


def load_forecasts(path: Path) -> list[Forecast]:
    """Read forecasts from a Parquet file.

    Args:
        path: The Parquet file read.

    Returns:
        list[Forecast]: The forecasts, in the order they were written.
    """
    table = pq.read_table(path, columns=FORECASTS_SCHEMA.names)
    table = table.unify_dictionaries().combine_chunks()
    activities = table['activities'].chunk(0)
    labels = np.array(activities.values.dictionary.to_pylist(), dtype=object)
    drawn = labels[activities.values.indices.to_numpy(zero_copy_only=False)]
    offsets = activities.offsets.to_numpy()
    suffixes = [tuple(drawn[a:b].tolist()) for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
    starts = np.flatnonzero(pc.equal(table['draw'], 0).to_numpy(zero_copy_only=False))
    stops = np.append(starts[1:], len(table))
    case_ids = table['case_id'].take(pa.array(starts)).to_pylist()
    prefix_lengths = table['prefix_length'].take(pa.array(starts)).to_pylist()
    return [
        Forecast(case_id=case_id, prefix_length=prefix_length, draws=suffixes[a:b])
        for case_id, prefix_length, a, b in zip(
            case_ids, prefix_lengths, starts, stops, strict=True
        )
    ]


class ForecastWriter:
    """Writes forecasts to a Parquet file a few at a time, so no forecast outlives its write."""

    def __init__(self, path: Path) -> None:
        """
        Args:
            path: The Parquet file written. Creates the parent directories if they do not exist.
        """
        self.path = path

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(self.path, schema=FORECASTS_SCHEMA)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._writer.close()

    def write(self, forecasts: Sequence[Forecast]) -> None:
        """Append forecasts to the file.

        Args:
            forecasts: The forecasts appended.
        """
        self._writer.write_table(_table(forecasts))


def _table(forecasts: Sequence[Forecast]) -> pa.Table:
    """The rows of `FORECASTS_SCHEMA` of the forecasts, one per draw.

    Args:
        forecasts: The forecasts.

    Returns:
        pa.Table: The rows of every draw, forecast after forecast.
    """
    draws = [draw for forecast in forecasts for draw in forecast.draws]
    activities = pa.array(list(chain.from_iterable(draws)), type=pa.string()).dictionary_encode()
    offsets = np.cumsum([0, *(len(draw) for draw in draws)], dtype=np.int32)
    columns = {
        'case_id': [forecast.case_id for forecast in forecasts for _ in forecast.draws],
        'prefix_length': [forecast.prefix_length for forecast in forecasts for _ in forecast.draws],
        'draw': [number for forecast in forecasts for number in range(len(forecast.draws))],
        'activities': pa.ListArray.from_arrays(
            offsets=pa.array(offsets),
            values=activities.cast(FORECASTS_SCHEMA.field('activities').type.value_type),
        ),
    }
    return pa.table(columns, schema=FORECASTS_SCHEMA)
