import numpy as np

from scipy.stats import gamma
from scipy.stats import rv_continuous


def discretise_gamma(a: float, scale: float, loc: float = 0) -> np.ndarray:
    """
    Returns a discretisation of a gamma distribution at values x = 0, 1, 2, 3, ..., ceiling(10^-6 quantile)
    """
    return discrete_distrb(gamma(a=a, scale=scale, loc=loc))


def discrete_distrb(distrb: rv_continuous) -> np.ndarray:
    """
    Returns a discretisation of specified distribution at values x = 0, 1, 2, 3, ..., ceiling(10^-6 quantile)
    """
    upper_lim = np.ceil(distrb.ppf(1 - 1e-6))
    bin_lims = np.linspace(0.5, upper_lim + 0.5, int(upper_lim + 1))
    cdf = distrb.cdf(bin_lims)
    pmf = np.diff(cdf, prepend=0)

    return pmf / pmf.sum()
