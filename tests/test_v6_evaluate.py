import tempfile, unittest
from pathlib import Path
import nibabel as nib
import numpy as np
from cardiac_phantom.v6_evaluate import sample
class EvaluationGeometryTests(unittest.TestCase):
 def test_physical_sampling_and_outside_mask(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'mask.nii.gz'; a=np.zeros((3,4,5),dtype='u1');a[1,2,3]=1;aff=np.diag([2.,3.,4.,1.]);aff[:3,3]=[10,-5,7];nib.save(nib.Nifti1Image(a,aff),path)
   xyz=nib.affines.apply_affine(aff,[[1,2,3],[8,8,8]]);result=sample(path,xyz,(2,),order=0);self.assertEqual(result[0],1);self.assertTrue(np.isnan(result[1]))
