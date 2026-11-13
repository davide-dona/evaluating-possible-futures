from enum import StrEnum


class Split(StrEnum):
    """The two splits preprocessing cuts a log into, named as their files on disk are."""

    TRAIN = 'train'
    TEST = 'test'


# Canonical column names, following the pm4py naming convention.
CASE_KEY = 'case:concept:name'
ACTIVITY_KEY = 'concept:name'
TIMESTAMP_KEY = 'time:timestamp'
# The lowest cut a case may be split at: 1, except on the test cases running across the
# train/test separation, whose earlier prefixes training could already have seen.
MIN_PREFIX_KEY = 'min_prefix_len'
# The highest cut a case may be split at: its length minus one, except on the cases running into
# the debiased end of the log, whose later prefixes are dropped.
MAX_PREFIX_KEY = 'max_prefix_len'
