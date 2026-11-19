import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager


def format_duration(seconds: float) -> str:
    """Format a duration as `4.2s`, `3m 12s` or `1h 05m`, whichever fits it.

    Args:
        seconds: The duration in seconds.

    Returns:
        str: The duration, in the largest unit that fits it.
    """
    if seconds < 60:
        return f'{seconds:.1f}s'
    minutes, seconds = divmod(int(seconds), 60)
    if minutes < 60:
        return f'{minutes}m {seconds:02d}s'
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h {minutes:02d}m'


def banner(title: str, fields: Mapping[str, object]) -> None:
    """Print the title of a stage and its settings, one aligned line each.

    Args:
        title: What the stage does.
        fields: The value of every setting, keyed by the name printed.
    """
    print(title, flush=True)
    width = max((len(name) for name in fields), default=0)
    for name, value in fields.items():
        print(f'  {name:<{width}}  {value}')
    print(flush=True)


@contextmanager
def step(message: str) -> Iterator[None]:
    """Print `message` before the block runs, then how long the block took.

    Args:
        message: What the block does.

    Returns:
        Iterator[None]: The context of the block.
    """
    print(f'{message}...', flush=True)
    started = time.perf_counter()
    yield
    print(f'  done in {format_duration(time.perf_counter() - started)}', flush=True)
