import unittest
from datetime import date

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal
from scipy.stats import gamma

from epyestim.estimate_r import overall_infectivity, sum_by_split_dates, estimate_r, gamma_quantiles


class EstimateRTest(unittest.TestCase):
    def test_overall_infectivity(self):

        infections_ts = pd.Series(
            [1, 3, 4, 7, 10, 3],
            index=pd.date_range(start='2020-01-01', end='2020-01-06')
        )
        gt_distribution = np.array([0.0, 0.3, 0.4, 0.2, 0.0])

        infectivity = overall_infectivity(infections_ts, gt_distribution)

        self.assertTrue(infectivity.index.equals(infections_ts.index))
        assert_array_almost_equal(
            np.array([0.0, 0.3, 1.3, 2.6, 4.3, 6.6]),
            infectivity.values
        )

    def test_split_dates(self):
        """
        day      1 2 3 4 5 6 7 8 9
        a        1 2 3 4 5 6 7 8 9
        b        9 8 7 6 5 4 3 2 1
        splits  ^-------^---------^
        sum(a)       10      35
        sum(b)       30      15
        """

        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'b': [9, 8, 7, 6, 5, 4, 3, 2, 1]
        }, index=pd.date_range(start='2020-01-01', end='2020-01-09'))

        split_dates = [date(2020, 1, 1), date(2020, 1, 5), date(2020,1, 10)]

        sums = sum_by_split_dates(df, split_dates)

        self.assertTrue(df.index.equals(sums.index))
        assert_array_almost_equal(np.array([10, 10, 10, 10, 35, 35, 35, 35, 35]), sums['a'])
        assert_array_almost_equal(np.array([30, 30, 30, 30, 15, 15, 15, 15, 15]), sums['b'])

    def test_estimate_r_rolling(self):
        infections_ts = pd.Series(
            [1, 3, 4, 7, 10, 3],
            index=pd.date_range(start='2020-01-01', end='2020-01-06')
        )
        gt_distribution = np.array([0.0, 0.3, 0.4, 0.2, 0.1])
        r_df = estimate_r(
            infections_ts=infections_ts,
            gt_distribution=gt_distribution,
            a_prior=1,
            b_prior=5,
            window_size=3,
        )

        self.assertTrue(r_df.index.equals(pd.date_range(start='2020-01-03', end='2020-01-06')))
        assert_array_almost_equal(np.array([9, 15, 22, 21]), r_df['a_posterior'])
        assert_array_almost_equal(np.array([0.555556, 0.227273, 0.117647, 0.070922]), r_df['b_posterior'])

    def test_estimate_r_boundary(self):
        infections_ts = pd.Series(
            [1, 3, 4, 7, 10, 3],
            index=pd.date_range(start='2020-01-01', end='2020-01-06')
        )
        gt_distribution = np.array([0.0, 0.3, 0.4, 0.2, 0.1])
        r_df = estimate_r(
            infections_ts=infections_ts,
            gt_distribution=gt_distribution,
            a_prior=1,
            b_prior=5,
            boundary_dates=[date(2020, 1, 1), date(2020, 1, 3), date(2020, 1, 6)]
        )

        self.assertTrue(r_df.index.equals(pd.date_range(start='2020-01-01', end='2020-01-05')))
        assert_array_almost_equal(np.array([5, 5, 22, 22, 22]), r_df['a_posterior'])
        assert_array_almost_equal(np.array([2, 2, 0.117647, 0.117647, 0.117647]), r_df['b_posterior'])

    def test_estimate_r_none_fail(self):
        infections_ts = pd.Series(
            [1, 3, 4, 7, 10, 3],
            index=pd.date_range(start='2020-01-01', end='2020-01-06')
        )
        gt_distribution = np.array([0.0, 0.3, 0.4, 0.2, 0.1])
        self.assertRaises(ValueError, lambda: estimate_r(
            infections_ts=infections_ts,
            gt_distribution=gt_distribution,
            a_prior=1,
            b_prior=5,
        ))

    def test_gamma_quantiles_equivalent(self):
        a = np.array([1.0, 2.0, 1.5, 2.5, 17.0, 13.0])
        b = np.array([2.0, 3.2, 5.1, 0.2, 34.6, 23.0])

        q = 0.3

        df = pd.DataFrame({'a_posterior': a, 'b_posterior': b})

        def get_r_quantile(q):
            def getter(row):
                return gamma(a=row['a_posterior'], scale=row['b_posterior']).ppf(q)

            return getter

        quantiles_slow = df.apply(get_r_quantile(q), axis=1)
        quantiles_fast = gamma_quantiles(q, df.a_posterior, df.b_posterior)

        assert_array_almost_equal(quantiles_slow, quantiles_fast)


if __name__ == '__main__':
    unittest.main()
