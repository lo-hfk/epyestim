from datetime import date
from typing import Optional, List, Iterable

import numpy as np
import pandas as pd

from scipy.stats import nbinom

from epyestim import bagging_r
from epyestim.distributions import discretise_gamma


def generate_standard_si_distribution():
    """
    Build the standard serial interval distribution
    """
    # Parameters used by [Flaxman et al., 2020]
    return discretise_gamma(6.5, scale=0.62)


def generate_standard_incubation_time_distribution():
    """
    Build the standard incubation time distribution
    """
    # Parameters used by [Brauner et al., 2020]
    return discretise_gamma(1.35, scale=3.77)


def generate_onset_to_reporting_distribution_singapore():
    """
    Build onset-to-reporting distribution
    """
    # Gamma fit for Singapore by [Tariq et al., 2020]
    return discretise_gamma(2, scale=3.2)


def generate_onset_to_reporting_distribution_brauner():
    """
    Build onset-to-reporting distribution
    """
    # Distribution used by [Brauner et al., 2020]
    mu = 5.25
    alpha = 1.57
    distrb = nbinom(n=1/alpha, p=1-alpha*mu/(1+alpha*mu))
    x = range(int(distrb.ppf(1 - 1e-6)))
    return distrb.pmf(x)


def generate_standard_infection_to_reporting_distribution():
    return np.convolve(generate_standard_incubation_time_distribution(),
                       generate_onset_to_reporting_distribution_brauner())


def r_covid(
        confirmed_cases: pd.Series,
        gt_distribution: np.array = generate_standard_si_distribution(),
        delay_distribution: np.array = generate_standard_infection_to_reporting_distribution(),
        a_prior: float = 3,
        b_prior: float = 1,
        smoothing_window: int = 21,
        r_window_size: Optional[int] = 3,
        r_interval_dates: Optional[List[date]] = None,
        n_samples: int = 100,
        quantiles: Iterable[float] = (0.025, 0.5, 0.975),
        auto_cutoff: bool = True
) -> pd.DataFrame:
    """
    Compute aggregated bootstrapped R and returns aggregate quantiles
    with default parameters for Covid-19.
    """
    return bagging_r(
        confirmed_cases,
        gt_distribution=gt_distribution,
        delay_distribution=delay_distribution,
        a_prior=a_prior,
        b_prior=b_prior,
        smoothing_window=smoothing_window,
        r_window_size=r_window_size,
        r_interval_dates=r_interval_dates,
        n_samples=n_samples,
        quantiles=quantiles,
        auto_cutoff=auto_cutoff
    )
