"""
This module implements estimation of the effective reproduction number from onset data
as developed by [Cori et al., 2013] and available in the R package 'epiestim'.
"""


from datetime import date
from typing import Optional, List

import numpy as np
import pandas as pd
from scipy.special import gammaincinv


def overall_infectivity(infections_ts: pd.Series, gt_distribution: np.array) -> pd.Series:

    padded_incidence = np.pad(
        infections_ts.values,
        (len(gt_distribution) - 1, 0),
        mode='constant', constant_values=0
    )

    infectivity_array = np.convolve(
        padded_incidence,
        gt_distribution,
        mode="valid"
    )

    return pd.Series(infectivity_array, index=infections_ts.index)


def estimate_r(
        infections_ts: pd.Series,
        gt_distribution: np.ndarray,
        a_prior: float,
        b_prior: float,
        window_size: Optional[int] = None,
        boundary_dates: Optional[List[date]] = None
) -> pd.DataFrame:
    """
    Estimate effective reproduction numbers using the Cori method.

    Either window_size or boundary_dates must be specified.

    :param infections_ts: time series of infection numbers
    :param gt_distribution: the generation time distribution
    :param a_prior: prior for the Gamma shape parameter for R
    :param b_prior: prior for the Gamma scale parameter for R
    :param window_size: size of the rolling window
    :param boundary_dates: boundaries of the intervals for which R should be estimated
    :return: dataframe with posterior values for shape (a) and scale (b) of the Gamma distribution for R
    """

    assert abs(gt_distribution.sum() - 1.0) < 0.001, "Serial interval distribution must sum to 1"
    assert (infections_ts >= 0).all(), "Infection numbers cannot be negative"

    if window_size is None and boundary_dates is None:
        raise ValueError("Either window_size or change_dates must be set")

    infectivity_ts = overall_infectivity(
        infections_ts=infections_ts,
        gt_distribution=np.asarray(gt_distribution)
    )

    df = pd.DataFrame(
        {'infections': infections_ts, 'infectivity': infectivity_ts}
    )

    sums = sum_by_split_dates(df, boundary_dates) if boundary_dates is not None else df.rolling(window_size).sum().dropna()

    return pd.DataFrame({
        'a_posterior': a_prior + sums.infections,
        'b_posterior': 1 / (1 / b_prior + sums.infectivity)
    })


def sum_by_split_dates(df: pd.DataFrame, split_dates: List[date]):
    idx = pd.cut(df.index, bins=np.array(split_dates, dtype=np.dtype('datetime64')), include_lowest=True, right=False)
    sums = df.groupby(idx).sum()
    return pd.DataFrame({'interval': idx}, index=df.index).join(sums, on='interval').drop('interval', axis=1).dropna()


def gamma_quantiles(q: float, a: np.ndarray, b: np.ndarray):
    return gammaincinv(a, q) * b
