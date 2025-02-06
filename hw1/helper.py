"""
EECS 445 - Introduction to Machine Learning
HW1 helper
"""

import pandas as pd
import numpy.typing as npt


def load_data(fname: str) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Loads the data in a csv file specified by `fname`.
    
    The file specified should be a csv with n rows and (d+1) columns, with the first column being the label.

    Args:
        fname: a string specifying the file to load.

    Returns:
        an nxd array X where n is the number of examples and d is the dimensionality and an nx1 array Y where
        n is the number of examples.
    """
    
    data = pd.read_csv(fname).values
    X = data[:, 1:]
    y = data[:, 0]
    return X, y
