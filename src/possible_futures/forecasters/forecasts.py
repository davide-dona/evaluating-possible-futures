from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import chain, pairwise
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from possible_futures.cases import Activities, Case
from possible_futures.forecasters.forecaster import Forecaster

# One row per draw, the draws of a prefix on consecutive rows and numbered from 0.
FORECASTS_SCHEMA = pa.schema(
    [
        ('case_id', pa.string()),
        ('prefix_length', pa.int64()),
        ('draw', pa.int32()),
        ('activities', pa.list_(pa.dictionary(pa.int16(), pa.string()))),
    ]
)


@dataclass(frozen=True, slots=True)
class Forecast:
    """The suffixes a forecaster drew for one prefix of a test case.

    Attributes:
        case_id: The case the prefix is taken from.
        prefix_length: The number of activities in the prefix.
        draws: The activities of every suffix drawn for the prefix.
    """

    case_id: str
    prefix_length: int
    draws: list[Activities]


def iter_forecasts(path: Path, *, max_rows: int) -> Iterator[list[Forecast]]:
    """Read forecasts from a Parquet file one chunk at a time, so memory holds one row group in
    Arrow's compact form and about `max_rows` draws as Python objects, plus the draws of one
    forecast that crosses the end of a chunk.

    Args:
        path: The Parquet file read.
        max_rows: The number of draws decoded at once.

    Yields:
        list[Forecast]: The next whole forecasts, in the order they were written.
    """
    file = pq.ParquetFile(path)
    carried: Forecast | None = None
    for group in range(file.num_row_groups):
        # `iter_batches` cannot decode the nested dictionary column, so read whole row groups
        # and slice them.
        table = file.read_row_group(group, columns=FORECASTS_SCHEMA.names)
        table = table.unify_dictionaries().combine_chunks()
        for batch in table.to_batches(max_chunksize=max_rows):
            continued, forecasts = _forecasts(batch)
            if carried is not None:
                forecasts.insert(0, replace(carried, draws=carried.draws + continued))
            # The last forecast of a batch may continue in the next one.
            carried = forecasts.pop()
            if forecasts:
                yield forecasts
    if carried is not None:
        yield [carried]


class ForecastWriter:
    """Appends forecasts to a Parquet file one batch at a time, so memory never holds more than
    one batch of draws. Use it as a context manager, which opens and closes the file.

    The batches go to a `.partial` file beside the forecasts file, which replaces it only once the
    block ends without an exception. A run that fails leaves the previous forecasts file, if any,
    untouched.
    """

    def __init__(self, path: Path) -> None:
        """
        Args:
            path: The Parquet file written. Creates the parent directories if they do not exist.
        """
        self.path = path
        self._partial = path.with_name(f'{path.name}.partial')

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(self._partial, schema=FORECASTS_SCHEMA)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._writer.close()
        if exc is None:
            self._partial.replace(self.path)
        else:
            self._partial.unlink()

    def write(self, forecasts: Sequence[Forecast]) -> None:
        """Append forecasts to the file.

        Args:
            forecasts: The forecasts appended.
        """
        self._writer.write_table(_table(forecasts))


def write_forecasts(
    forecaster: Forecaster,
    *,
    test: Sequence[Case],
    scored: Mapping[str, range],
    path: Path,
    num_draws: int,
    max_steps: int,
    max_rows: int,
) -> int:
    """Draw suffixes for every scored prefix of the test cases and save them to `path`.

    Args:
        forecaster: The forecaster the suffixes are drawn from.
        test: The test cases.
        scored: The lengths of the scored prefixes of every test case, keyed by case id. A case
            absent from it has none.
        path: The Parquet file the forecasts are written to.
        num_draws: The number of suffixes drawn for each prefix.
        max_steps: The most activities a suffix holds.
        max_rows: The most suffixes drawn at once.

    Returns:
        int: The number of prefixes written.
    """
    # Shortest first, so prefixes of similar length share a batch and little of it is padding.
    prefixes = sorted(
        ((case, length) for case in test for length in scored.get(case.case_id, ())),
        key=lambda prefix: prefix[1],
    )
    batch_size = max(1, max_rows // num_draws)
    with ForecastWriter(path) as writer:
        for start in range(0, len(prefixes), batch_size):
            batch = prefixes[start : start + batch_size]
            draws = forecaster.draw(
                [case.prefix(length) for case, length in batch],
                num_draws=num_draws,
                max_steps=max_steps,
            )
            writer.write(
                [
                    Forecast(case_id=case.case_id, prefix_length=length, draws=suffixes)
                    for (case, length), suffixes in zip(batch, draws, strict=True)
                ]
            )
    return len(prefixes)


def _forecasts(batch: pa.RecordBatch) -> tuple[list[Activities], list[Forecast]]:
    """Decode the rows of `FORECASTS_SCHEMA` of one batch read from a forecasts file.

    Args:
        batch: Consecutive rows of the file.

    Returns:
        The draws of the rows before the first draw numbered 0, which continue the forecast of
        the previous batch, and the forecasts that start in the batch. The last of them may
        continue in the next batch.
    """
    activities = batch.column('activities')
    values = activities.flatten()  # Only the activities of this batch, unlike `.values`
    labels = np.array(values.dictionary.to_pylist(), dtype=object)
    drawn = labels[values.indices.to_numpy(zero_copy_only=False)]
    offsets = activities.offsets.to_numpy()
    offsets = offsets - offsets[0]
    suffixes = [tuple(drawn[a:b].tolist()) for a, b in pairwise(offsets)]
    starts = np.flatnonzero(pc.equal(batch.column('draw'), 0).to_numpy(zero_copy_only=False))
    # Every forecast spans from its start to the next one, the last to the end of the batch.
    bounds = np.append(starts, batch.num_rows).tolist()
    case_ids = batch.column('case_id').take(pa.array(starts)).to_pylist()
    prefix_lengths = batch.column('prefix_length').take(pa.array(starts)).to_pylist()
    forecasts = [
        Forecast(case_id=case_id, prefix_length=prefix_length, draws=suffixes[a:b])
        for case_id, prefix_length, (a, b) in zip(
            case_ids, prefix_lengths, pairwise(bounds), strict=True
        )
    ]
    return suffixes[: bounds[0]], forecasts


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
