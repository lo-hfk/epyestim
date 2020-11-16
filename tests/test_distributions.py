import unittest

from epyestim.distributions import discretise_gamma


class DistributionsTest(unittest.TestCase):
    def test_si_distribution(self):
        self.assertAlmostEqual(1.0, discretise_gamma(a=5, scale=10).sum(), 10)

