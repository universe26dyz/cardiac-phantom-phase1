import unittest
import numpy as np
from cardiac_phantom import Phantom
from cardiac_phantom.motion import Motion
class MotionTests(unittest.TestCase):
    def test_roundtrip_period_pose(self):
        a=.42; p=Phantom(np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]]),np.array([3,8,-2])); m=Motion(p)
        xyz=np.random.default_rng(11).uniform([-45,-35,-60],[60,35,60],(4000,3))
        for t in (0,.23,.5,1.,4.):
            world=m.canonical_to_world(xyz,t); np.testing.assert_allclose(m.world_to_canonical(world,t),xyz,atol=1e-9)
            for key in ('label','t1_ms','t2_ms'): np.testing.assert_allclose(m.query(world,t)[key],p.query(p.local_to_world(xyz))[key],atol=1e-9,equal_nan=True)
        np.testing.assert_allclose(m.canonical_to_world(xyz,0),p.local_to_world(xyz),atol=1e-12)
        np.testing.assert_allclose(m.canonical_to_world(xyz,4),p.local_to_world(xyz),atol=1e-12)
        np.testing.assert_allclose(m.canonical_to_world(xyz,.2,False,True),m.canonical_to_world(xyz,.2,cardiac=False),atol=1e-12)
        np.testing.assert_allclose(m.canonical_to_world(xyz,.2,True,False),m.canonical_to_world(xyz,.2,respiratory=False),atol=1e-12)
    def test_jacobian_and_interface(self):
        m=Motion(); pts=np.random.default_rng(7).uniform([-40,-30,-50],[50,30,50],(1000,3)); eps=1e-4
        for t in (.25,.5,.75):
            jac=np.stack([(m.canonical_to_world(pts+np.eye(3)[i]*eps,t)-m.canonical_to_world(pts-np.eye(3)[i]*eps,t))/(2*eps) for i in range(3)],axis=-1)
            self.assertGreater(np.linalg.det(jac).min(),0)
            p=np.array([[31.99,0,0],[32.01,0,0]]); np.testing.assert_array_equal(m.query(m.canonical_to_world(p,t),t)['label'],[1,3])
    def test_world_changes(self):
        m=Motion(); self.assertNotEqual(m.query([20,0,0],0)['label'],m.query([20,0,0],.5,True,False)['label'])
