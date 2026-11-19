import tempfile, unittest
import numpy as np
import nibabel as nib
from cardiac_phantom import Phantom, sample_plane
from cardiac_phantom.export import grid, sample_grid, save_maps

class FixTests(unittest.TestCase):
    def test_interface(self):
        p=Phantom(amplitude=0); z=12.37
        for y in (-10.,0.,10.):
            x=32*np.sqrt(1-(y/28)**2-(z/55)**2); point=np.array([x,y,z]); n=point/np.array([32**2,28**2,55**2]); n/=np.linalg.norm(n)
            np.testing.assert_array_equal(p.query(np.array([point-.01*n,point+.01*n]))['label'],[1,3])
        self.assertEqual(p.query([32.,0.,0.])['label'],1)
    def test_texture(self):
        p=Phantom(); rng=np.random.default_rng(4); xyz=rng.uniform([-33,-29,-50],[55,29,50],(4000,3)); q=p.query(xyz)
        other=Phantom(seed=999).query(xyz); reverse=p.query(xyz[::-1])
        for k in (1,2,3):
            mask=q['label']==k
            for j,name in enumerate(('t1_ms','t2_ms')):
                v=q[name][mask]/p.tissues[k][j]; self.assertTrue(np.all((v>=.95)&(v<=1.05))); self.assertGreater(np.std(v),0); self.assertFalse(np.array_equal(q[name][mask],other[name][mask])); np.testing.assert_allclose(q[name],reverse[name][::-1],equal_nan=True)
        np.testing.assert_array_equal(q['label'],other['label'])
        a=.4; R=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
        transformed=Phantom(R,np.array([3,9,-4])); t=transformed.query(transformed.local_to_world(xyz)); np.testing.assert_allclose(q['t1_ms'],t['t1_ms'],equal_nan=True)
        one=p.query(xyz[:100]); two=p.query(xyz[:50]); np.testing.assert_allclose(one['t2_ms'][:50],two['t2_ms'],equal_nan=True)
    def test_nifti_and_grid(self):
        p=Phantom(); shape,aff=grid(2); maps=sample_grid(p.query,shape,aff)
        with tempfile.TemporaryDirectory() as d:
            save_maps(d+'/static',maps,aff)
            for k,v in maps.items():
                img=nib.load(d+'/static/'+k+'.nii.gz'); self.assertEqual(img.shape,shape); np.testing.assert_allclose(img.affine,aff); self.assertEqual(img.header.get_xyzt_units()[0],'mm'); self.assertEqual(img.get_data_dtype(),v.dtype)
                np.testing.assert_allclose(np.asarray(img.dataobj)[25,25,35],p.query([0,0,0])[k],rtol=1e-6,equal_nan=True)
        s1,a1=grid(1); np.testing.assert_allclose(a1@np.array([50,50,70,1]),aff@np.array([25,25,35,1]))
