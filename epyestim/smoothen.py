import numpy as np
import pandas as pd

import warnings

from statsmodels.nonparametric.smoothers_lowess import lowess


def renormalise_series(ser: pd.Series, reference_ser: pd.Series) -> pd.Series:
    """
    Renormalises time series ser with respect to series reference_ser

    :param ser: (pd.Series): series to be renormalised (e.g. smoothed version of reference_ser)
    :param reference_ser: (pd.Series): reference series of same length (e.g. ser before smoothing)
    """

    assert len(ser.index) == len(reference_ser.index), \
        'length of series to be renormalised does not match length of reference series'

    ser[ser < 0] = 0

    norm_ser = ser.sum()
    norm_ref_ser = reference_ser.sum()

    ser = ser * norm_ref_ser / norm_ser

    return ser


def smoothen_series(ser: pd.Series, window_width: int) -> pd.Series:
    """
    Smoothen the time series with LOWESS, using standard tricubic weights.

    :param ser: (pd.Series): time series (index must be consecutive dates)
    :param window_width: (int): width of moving time frame (in days) for local regression
    """

    if window_width != 1:
        if window_width % 7 != 0:
            warnings.warn('window_width is recommended to be a multiple of 7 to account for weekly patterns.')

    endog = ser.values
    exog = np.array(range(len(ser)))
    frac = window_width / len(exog)

    smooth = lowess(endog=endog, exog=exog, frac=frac, it=0, delta=0.0, is_sorted=True, missing='raise',
                    return_sorted=False)
    smooth_ser = pd.Series(smooth, index=ser.index)
    smooth_ser = renormalise_series(smooth_ser, ser)

    return smooth_ser
