from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd


def split_proportionally(
    df: pd.DataFrame, by: Union[str, Tuple[str, ...]], proportions: Dict[str, float]
) -> Dict[str, pd.DataFrame]:
    """Split a dataframe into a few dataframes, containing given proportions of every row group.

    Returns a dictionary with the same keys as `proportions` and values consisting of dataframes
    that partition `df`. Each part contains roughly `proportions[key]` of the rows of each group
    in `df.groupby(by)` (the numbers are rounded in order to always return a partition).

    Example:
        datasets = split_proportionally(df, "class", {"train": 0.5, "val": 0.25, "test": 0.25})
    """
    thresholds = np.asarray(list(proportions.values())).cumsum()
    if not np.isclose(thresholds[-1], 1):
        raise ValueError(f"Proportions should sum to 1, got: {proportions}")
    thresholds[-1] = 1.0

    grouped = df.groupby(by)[df.columns[0]]  # Pick an arbitrary column
    index_in_group = grouped.cumcount()
    group_len = grouped.transform(len)

    result: Dict[str, pd.DataFrame] = {}
    done_mask = index_in_group < 0  # Initially all False.
    for key, threshold in zip(proportions.keys(), thresholds):
        mask = ~done_mask & (index_in_group < threshold * group_len)
        result[key] = df[mask]
        done_mask = done_mask | mask
    assert done_mask.all()
    return result
