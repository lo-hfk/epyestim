import pandas as pd
import numpy as np


def bootstrap_series(ser: pd.Series) -> pd.Series:
    """
    Creates a bootstrapped time series of same length and number of observations as original time series

    :param ser: (pd.Series): time series (not necessarily stationary, but observations
                are assumed to be weakly correlated.)
    """
    resampled = np.random.multinomial(ser.sum(), ser / ser.sum())
    return pd.Series(resampled, index=ser.index)
