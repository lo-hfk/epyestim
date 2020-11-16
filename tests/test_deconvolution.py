import unittest

import pandas as pd
import numpy as np
from numpy.testing import assert_array_almost_equal

from epyestim.deconvolution import deconvolve_series


class DeconvolutionTest(unittest.TestCase):
    def test_deconvolve_series(self):
        ser = pd.Series(
            [1, 2],
            index=pd.date_range(start='2020-04-01', periods=2)
        )
        delay_distrb = np.array([0.25, 0.5, 0.25])

        ser_deconvolved = deconvolve_series(ser, delay_distrb)

        self.assertTrue(ser_deconvolved.index.equals(pd.date_range(start='2020-03-30', periods=4)))
        assert_array_almost_equal(
            np.array([4/5, 32/35, 72/35, 16/7]),
            ser_deconvolved.values
        )


if __name__ == '__main__':
    unittest.main()
