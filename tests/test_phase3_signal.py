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

class MatlabBloodParityTests(unittest.TestCase):
    def test_heart_and_blood_multiple_timing(self):
        cases = [
            (1200,45,900,[.201925296507001,.0991636342780179,.161643624883759,.174609060763169,.0741329931423085,.125682250863803,.167146593428093,.0970307616624107,.0783176585946209,.0677138530794781]),
            (1600,180,1000,[.29099236657308,.039186384230249,.176797775955806,.234151240061449,.162246685236168,.0928448112014641,.199161452586564,.203482131077473,.194918517199744,.182688387086262]),
            (1520,171,1150,[.286606475292138,.0793403348035483,.211854432924556,.259839510854959,.177057036765468,.119009610899243,.226219193594446,.219861155037964,.206793355286021,.192744695251853]),
            (1680,189,800,[.294996288194207,.0133927315110999,.12392108766149,.190877219742523,.131708451852401,.0662286933622853,.162745604177207,.176412269609576,.173422876107837,.164129048996149]),
        ]
        for t1,t2,rr,expected in cases:
            actual=fingerprints(np.array([t1]),np.array([t2]),timing9(np.arange(10)*rr))[0]
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-6)
