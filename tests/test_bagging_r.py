import unittest

import numpy as np
import pandas as pd

from epyestim import bagging_r


class BaggingRTest(unittest.TestCase):
    def test_bagging_r_simple(self):
        confirmed_cases = pd.Series(
            np.floor(np.exp(np.arange(50) / 10)),
            index=pd.date_range(start='2020-03-01', periods=50)
        )
        gt_distribution = np.array([0.25, 0.5, 0.25])
        delay_distribution = np.array([0.1, 0.5, 0.4])

        bag = bagging_r(
            confirmed_cases=confirmed_cases,
            gt_distribution=gt_distribution,
            delay_distribution=delay_distribution,
            a_prior=1,
            b_prior=3,
            n_samples=100,
            smoothing_window=14,
            r_window_size=1,
            quantiles=[0.3, 0.7]
        )

        self.assertSetEqual({'Q0.3', 'Q0.7', 'R_mean', 'R_var', 'cases'}, set(bag.columns))


if __name__ == '__main__':
    unittest.main()
