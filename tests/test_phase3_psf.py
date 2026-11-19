import unittest
import numpy as np
from cardiac_phantom.phase3.psf import integrate, add_noise
from cardiac_phantom.phase3.signal import fingerprints, timing9

class PsfNoiseTests(unittest.TestCase):
    def test_constant_and_boundary_in_signal_domain(self):
        timing = timing9(np.arange(10)*1000)
        values = fingerprints(np.array([1200.,1600.]),np.array([45.,180.]),timing)
        np.testing.assert_allclose(integrate(np.stack([values[0]]*3),[1/3]*3),values[0])
        actual = integrate(np.stack([values[0],values[1],np.zeros(10)]),[.2,.3,.5])
        np.testing.assert_allclose(actual, .2*values[0]+.3*values[1])
        averaged = fingerprints(np.array([1440.]),np.array([126.]),timing)[0]*.5
        self.assertGreater(np.max(np.abs(actual-averaged)),1e-3)
        with self.assertRaises(ValueError): integrate(np.ones((3,10)),[.5,.5,.5])
    def test_noise(self):
        clean=np.ones(200000); a=add_noise(clean,.02,42)
        np.testing.assert_array_equal(a,add_noise(clean,.02,42))
        np.testing.assert_array_equal(clean,add_noise(clean,0,42))
        self.assertLess(abs((a-clean).mean()),.0002)
        self.assertLess(abs((a-clean).std()-.02),.0002)
