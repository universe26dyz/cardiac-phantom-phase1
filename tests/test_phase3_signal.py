import unittest
import numpy as np
from cardiac_phantom.phase3.signal import fingerprints, timing9

class SignalTests(unittest.TestCase):
    def test_timing_contract(self):
        np.testing.assert_allclose(timing9(np.arange(10)*1000.), [746.83,746.83,746.83,646.83,746.83,746.83,711.83,701.83,691.83])
        with self.assertRaises(ValueError): timing9(np.arange(10)*100.)
    def test_frozen_matlab_reference(self):
        expected=[.201925296507001,.112896941067627,.17300329171639,.184478889552632,.0835032590914123,.135506260410118,.177319898087017,.101549088567306,.081997893938223,.0707318204500697]
        actual=fingerprints(np.array([1200.]),np.array([45.]),timing9(np.arange(10)*1000.))[0]
        np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=0)
    def test_background_rejected(self):
        with self.assertRaises(ValueError): fingerprints(np.array([np.nan]),np.array([np.nan]),timing9(np.arange(10)*1000.))
