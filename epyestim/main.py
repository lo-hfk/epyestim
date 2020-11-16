from datetime import date
from datetime import timedelta

import numpy as np
import pandas as pd

import warnings

from typing import List, Optional, Iterable

from . import bootstrap
from . import smoothen
from . import deconvolution
from . import estimate_r
from .estimate_r import gamma_quantiles


def sample_r(
        confirmed_cases: pd.Series,
        gt_distribution: np.ndarray,
        delay_distribution: np.ndarray,
        a_prior: float,
        b_prior: float,
        smoothing_window: int,
        r_window_size: Optional[int] = None,
        r_interval_dates: Optional[List[date]] = None,
) -> pd.DataFrame:
    """
    Sample R once.
    """
    bs_confirmed_cases = bootstrap.bootstrap_series(confirmed_cases)
    sm_bs_confirmed_cases = smoothen.smoothen_series(bs_confirmed_cases, smoothing_window)
    max_likelihood_infections = deconvolution.deconvolve_series(sm_bs_confirmed_cases, delay_distribution)
    posterior_reproduction_number = estimate_r.estimate_r(
        infections_ts=max_likelihood_infections,
        gt_distribution=gt_distribution,
        a_prior=a_prior,
        b_prior=b_prior,
        window_size=r_window_size,
        boundary_dates=r_interval_dates
    )
    assert len(posterior_reproduction_number) > 0
    return posterior_reproduction_number


def aggregate_quantiles_r(bag: List[pd.DataFrame], quantiles: Iterable[float]) -> pd.DataFrame:
    """
    Compute median quantiles of many samples of R
    """
    all_samples = pd.concat([r_sample for r_sample in bag], axis=0)
    all_samples['R_mean'] = all_samples.a_posterior * all_samples.b_posterior
    all_samples['R_var'] = all_samples.a_posterior * all_samples.b_posterior ** 2
    for q in quantiles:
        all_samples[f'Q{q}'] = gamma_quantiles(q, all_samples.a_posterior, all_samples.b_posterior)
    aggregate_quantiles = all_samples.drop(['a_posterior', 'b_posterior'], axis=1).groupby(all_samples.index).median()
    return aggregate_quantiles


def start_date(
        confirmed_cases: pd.Series,
        gt_distribution: np.ndarray,
        r_window_size: Optional[int] = None,
) -> date:
    """
    determine start date of reliable R-estimation: 3 conditions to be fulfilled according to [Cori et al., 2013]
        condition 1: cumulative cases have reached at least 12
        condition 2: at least one mean generation time after index case
        condition 3: if applicable: at least one r_window_size after index case
    """
    cumulative_cases = confirmed_cases.cumsum()
    over_dozen_cases_date = cumulative_cases.ge(12).idxmax()
    index_case_date = confirmed_cases.ne(0).idxmax()
    mean_gt = gt_distribution @ np.arange(len(gt_distribution)) / gt_distribution.sum()

    r_window_size = r_window_size if isinstance(r_window_size, int) else 0

    condition1_date = over_dozen_cases_date
    condition2_date = index_case_date + timedelta(days=np.ceil(mean_gt))
    condition3_date = index_case_date + timedelta(days=r_window_size)

    first_true_date = max(condition1_date, condition2_date, condition3_date)

    return first_true_date


def end_date(confirmed_cases: pd.Series, delay_distribution: np.ndarray) -> date:

    delay_mean = delay_distribution.dot(np.arange(len(delay_distribution)))
    return confirmed_cases.index[-1] - timedelta(days=int(delay_mean))


def bagging_r(
        confirmed_cases: pd.Series,
        gt_distribution: np.ndarray,
        delay_distribution: np.ndarray,
        a_prior: float,
        b_prior: float,
        smoothing_window: int,
        r_window_size: Optional[int] = None,
        r_interval_dates: Optional[List[date]] = None,
        n_samples: int = 100,
        quantiles: Iterable[float] = (0.025, 0.5, 0.975),
        auto_cutoff: bool = True
) -> pd.DataFrame:
    """
    Compute aggregated bootstrapped R and returns aggregate quantiles
    """

    # check and modify user input
    assert not any(q >= 1 or q <= 0 for q in quantiles), 'quantiles must be between 0 and 1'
    assert isinstance(confirmed_cases, pd.Series), 'confirmed cases must be of type pandas.Series'
    confirmed_cases = confirmed_cases.astype(int)
    if r_interval_dates is not None:
        r_interval_dates = sorted(r_interval_dates, reverse=False)
    if auto_cutoff:
        start_cutoff_date = start_date(
            confirmed_cases=confirmed_cases,
            gt_distribution=gt_distribution,
            r_window_size=r_window_size
        )
        end_cutoff_date = end_date(
            confirmed_cases=confirmed_cases,
            delay_distribution=delay_distribution
        )
        if r_interval_dates is not None:
            if r_interval_dates[0] < start_cutoff_date:
                r_interval_dates = [start_cutoff_date] + [d for d in r_interval_dates if d > start_cutoff_date]
                warnings.warn(
                    f'First interval start reset to {str(start_cutoff_date)}. '
                    'If you do not want that, set auto_cutoff=False.'
                )
            if r_interval_dates[-1] > end_cutoff_date:
                r_interval_dates = [d for d in r_interval_dates if d < end_cutoff_date] + [end_cutoff_date]
                warnings.warn(
                    f'First interval end reset to {str(end_cutoff_date)}. '
                    'If you do not want that, set auto_cutoff=False.'
                )

    if r_interval_dates is not None:
        assert len(r_interval_dates) > 1, 'There are less than two interval boundaries.'

    # bootstrap bagging
    bag = []
    for _ in range(n_samples):
        posterior_r = sample_r(
            confirmed_cases=confirmed_cases,
            gt_distribution=gt_distribution,
            delay_distribution=delay_distribution,
            a_prior=a_prior,
            b_prior=b_prior,
            smoothing_window=smoothing_window,
            r_window_size=r_window_size,
            r_interval_dates=r_interval_dates
        )
        bag.append(posterior_r)
    r_output = aggregate_quantiles_r(bag, quantiles)
    r_output = pd.concat((confirmed_cases.rename('cases'), r_output), axis=1)

    # do auto-cutoff
    if auto_cutoff:
        r_output = r_output[(start_cutoff_date <= r_output.index) & (r_output.index <= end_cutoff_date)]

    return r_output
