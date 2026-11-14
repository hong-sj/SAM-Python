"""
Example dataset loaders for Shared Anchor Matching.
"""

from importlib.resources import files

import pandas as pd


def load_sample_3group():
    """
    Load the three-treatment-group example dataset.

    Returns
    -------
    pandas.DataFrame
        Three-treatment-group example data.
    """
    path = files("samatch").joinpath("data", "sample_3group.csv")

    with path.open("r", encoding="utf-8") as file:
        return pd.read_csv(file)


def load_sample_4group():
    """
    Load the four-treatment-group example dataset.

    Returns
    -------
    pandas.DataFrame
        Four-treatment-group example data.
    """
    path = files("samatch").joinpath("data", "sample_4group.csv")

    with path.open("r", encoding="utf-8") as file:
        return pd.read_csv(file)