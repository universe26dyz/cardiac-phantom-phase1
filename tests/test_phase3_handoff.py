import unittest
import numpy as np

class GeometryLandmarkTests(unittest.TestCase):
    def test_4ch_row_col_and_normal_orientation(self):
        row=np.array([0.,0.,1.]); col=np.array([1.,0.,0.]); sampling_normal=np.array([0.,1.,0.]); center=np.array([0.,7.,0.]); spacing=5.; thickness=8.; ras_lps=np.diag([-1.,-1.,1.]); proper_normal=np.cross(col,row)
        self.assertEqual(float(proper_normal @ sampling_normal),-1.)
        a=np.eye(4);a[:3,0]=ras_lps@(spacing*row);a[:3,1]=ras_lps@(spacing*col);a[:3,2]=ras_lps@(thickness*proper_normal);a[:3,3]=ras_lps@(center-15.5*spacing*(row+col))
        index=np.array([7.,19.,0.,1.]);world=center+(7-15.5)*spacing*row+(19-15.5)*spacing*col
        np.testing.assert_allclose(ras_lps@(a@index)[:3],world);np.testing.assert_allclose(np.linalg.inv(a)@np.r_[ras_lps@world,1.],index)
