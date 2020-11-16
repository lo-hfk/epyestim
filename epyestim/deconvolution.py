import numpy as np
import datetime as dt
import pandas as pd

import scipy.linalg as la


def shift_and_pad(ser: pd.Series, delay_distrb: np.ndarray) -> pd.Series:
    """
    Shift the time series of reported cases backwards to generate an initial guess for the deconvolution iteration.

    :param ser: (pd.Series): time series of reported cases (index must be consecutive dates)
    :param delay_distrb: (1-dimensional np.ndarray): discretised delay distribution (delay_distrb[j] = probability
                         that an infection gets detected with a delay of j days). Array must contain no zero values.
    """

    assert not (delay_distrb == 0).any(), 'Delay distribution array must contain only nonzero values'

    # extend time series backwards and add padding
    k = len(delay_distrb)  # k-1 = maximum days from infection to reporting (according to discretised distribution)
    ix_start_new = ser.index[0] - dt.timedelta(days=k - 1)
    ix_end = ser.index[-1]
    new_ix = pd.date_range(start=ix_start_new, end=ix_end, freq='D')
    ser_extended = ser.reindex(new_ix, method='bfill')

    # shift time series
    mode_delay = delay_distrb.argmax()  # the mode of the discretised delay distribution
    ser_shifted = ser_extended.shift(-mode_delay).ffill()

    # replace non-positive values with ones
    ser_shifted[ser_shifted <= 0] = 1

    return ser_shifted


def deconvolve_series(ser: pd.Series, delay_distrb: np.ndarray) -> pd.Series:
    """
    Get maximum-likelihood time series of infections, given a time series of reported cases
    and discrete distribution  of delays from infection to reporting.

    :param ser: (pd.Series): time series of reported cases (index must be consecutive dates)
    :param delay_distrb: (1-dimensional np.ndarray): discretised delay distribution (delay_distrb[j] = probability
                         that an infection gets detected with a delay of j days)
    """

    assert abs(delay_distrb.sum() - 1.0) < 0.001, "Delay distribution must sum to 1"

    # cut series before first non-vanishing value
    assert all(ser >= 0)
    ser = ser[ser.ne(0).idxmax():]

    # get initial value for iteration
    ser_deconvolved = shift_and_pad(ser, delay_distrb)

    # parameters
    k = len(delay_distrb)  # k-1 = maximum days from infection to reporting (according to discretised distribution)
    n = len(ser_deconvolved)

    # delay kernel
    delay_distrb_rev = np.flip(delay_distrb)
    col = np.concatenate(([delay_distrb_rev[0]], np.zeros(n - k)), axis=None)
    row = np.concatenate((delay_distrb_rev, np.zeros(n - k)), axis=None)
    delay_kernel = np.concatenate((np.zeros((k - 1, n)), la.toeplitz(col, row)), axis=0)
        # 2d numpy ndarray of dimension (n,n)
        # delay_distrb[i,j] := delay_distrb[i-j]      for i=k,...n   j=1,...,n
        # delay_distrb[i,j] := 0                      for i=1,...,k-1   j=1,...,n

    q = pd.Series(np.sum(delay_kernel, axis=0), index=ser_deconvolved.index)
        # pandas time series from day 1 to n,
        # q_j = probability that infection on day j gets reported in the time window between day k and n.   j=1,...,n

    # Richardson-Lucy deconvolution iteration
    for _ in range(100):
        exp_obs_array = delay_kernel.dot(ser_deconvolved)
        expected_observed = pd.Series(exp_obs_array[(k - 1):], index=ser.index)
        mD_overE = delay_kernel[(k - 1):, :] * (ser / expected_observed).values.reshape(-1, 1)
        ser_deconvolved = (ser_deconvolved / q) * np.sum(mD_overE, axis=0)
        assert not ser_deconvolved.isna().any(), f'deconvolution has nan in iteration {_}'
        EminusDsq_overE = np.square(expected_observed - ser)
        chi_sq = (EminusDsq_overE / expected_observed).sum() / (n - k + 1)
        if chi_sq < 1:
            break

    return ser_deconvolved
