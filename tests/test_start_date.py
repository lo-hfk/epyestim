import unittest

from datetime import date

import pandas as pd
import numpy as np

from epyestim.main import start_date


class StartDateTest(unittest.TestCase):
    def test_start_date1(self):
        confirmed_cases = pd.Series(
            [0, 0, 0, 0, 2, 0, 0, 3, 0, 1, 1, 4, 2, 1, 8, 2, 5, 10, 9, 6],
            index=pd.date_range(start='2020-03-01', periods=20)
        )
        gt_distribution = np.array([0.2, 0.3, 0.2, 0.1, 0.1, 0.1])

        date1 = start_date(confirmed_cases, gt_distribution=gt_distribution)

        self.assertEqual(date(2020, 3, 13), date1)

    def test_start_date2(self):
        confirmed_cases = pd.Series(
            [0, 0, 0, 0, 2, 0, 0, 3, 0, 1, 1, 4, 2, 1, 8, 2, 5, 10, 9, 6],
            index=pd.date_range(start='2020-03-01', periods=20)
        )
        gt_distribution = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.05, 0.05,
                                    0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])

        date2 = start_date(confirmed_cases, gt_distribution=gt_distribution)

        self.assertEqual(date(2020, 3, 17), date2)

    def test_start_date3(self):
        confirmed_cases = pd.Series(
            [0, 0, 0, 0, 2, 0, 0, 3, 0, 1, 1, 4, 2, 1, 8, 2, 5, 10, 9, 6],
            index=pd.date_range(start='2020-03-01', periods=20)
        )
        gt_distribution = np.array([0.2, 0.3, 0.2, 0.1, 0.1, 0.1])

        date3 = start_date(confirmed_cases, gt_distribution=gt_distribution, r_window_size=10)

        self.assertEqual(date(2020, 3, 15), date3)


if __name__ == '__main__':
    unittest.main()