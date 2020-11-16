import unittest

import pandas as pd
import numpy as np
from numpy.testing import assert_array_almost_equal

from epyestim.smoothen import renormalise_series, smoothen_series


class MyTestCase(unittest.TestCase):
    def test_renormalise_series(self):
        ser = pd.Series([-1,2,3], pd.date_range('2020-03-01', periods=3))
        reference_ser = pd.Series([1,2,4], pd.date_range('2020-03-01', periods=3))

        renormalised_ser = renormalise_series(ser, reference_ser)
        expected_ser = pd.Series(7*np.array([0,2,3])/5, pd.date_range('2020-03-01', periods=3))

        assert_array_almost_equal(renormalised_ser, expected_ser)

    def test_smoothen_series_donothing(self):
        ser = pd.Series([1, 2, 3], pd.date_range('2020-03-01', periods=3))
        smooth_ser = smoothen_series(ser, window_width=1)
        assert_array_almost_equal(smooth_ser, ser)