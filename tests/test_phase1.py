import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from cardiac_phantom import Phantom, sample_plane

class TestPhase1(unittest.TestCase):
    def setUp(self): self.p = Phantom(amplitude=0)
    def test_anatomy(self):
        pts = [[0,0,0], [26,0,0], [45,0,0], [90,0,0], [0,0,-50], [0,0,52], [0,0,-56], [0,0,56]]
        np.testing.assert_array_equal(self.p.query(pts)['label'], [2,1,3,0,1,1,0,0])
        x,y,z = np.meshgrid(np.arange(-60,61,2),np.arange(-60,61,2),np.arange(-60,61,2), indexing='ij')
        f = self.p.fields(np.stack([x,y,z], axis=-1))
        self.assertFalse(np.any((f['lv_inner'] <= 0) & (f['lv_outer'] > 0)))
        self.assertFalse(np.any((f['rv'] < 0) & (f['lv_outer'] <= 0)))
        # A center radial line on the middle SAX passes through blood, myocardium, background.
        np.testing.assert_array_equal(self.p.query([[0,0,12.37],[25,0,12.37],[0,35,12.37]])['label'], [2,1,0])
        from scipy.ndimage import label
        labels = self.p.query(np.stack([x,y,z], axis=-1))['label']
        for k in (1,2,3): self.assertEqual(label(labels == k)[1], 1)
    def test_transform_planes_and_continuity(self):
        a = .47; R = np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
        p = Phantom(R, np.array([10,-2,3])); pts = np.array([[0,0,0],[4,6,-3]])
        np.testing.assert_allclose(p.world_to_local(p.local_to_world(pts)), pts, atol=1e-12)
        for center, col, row in [([0,0,12.37],[1,0,0],[0,1,0]), ([0,0,5],[2**-.5,2**-.5,0],[-.5,.5,2**-.5])]:
            xyz, q = sample_plane(p, center, col, row, (1.3, .7), (11,13))
            for i,j in [(0,0),(5,6),(10,12)]:
                self.assertEqual(q['label'][i,j], p.query(xyz[i,j])['label'])
            self.assertEqual(xyz.shape, (11,13,3))
        z = np.array([12.36,12.37,12.38]); boundary = 22*np.sqrt(1-((z-2)/45)**2)
        np.testing.assert_array_equal(self.p.query(np.stack([boundary-.001, z*0, z], axis=-1))['label'], [2,2,2])
        np.testing.assert_array_equal(self.p.query(np.stack([boundary+.001, z*0, z], axis=-1))['label'], [1,1,1])
        with self.assertRaises(ValueError): sample_plane(p,[0,0,0],[1,0,0],[1,0,0],(1,1),(5,5))
    def test_parameters(self):
        pts = [[90,0,0],[26,0,0],[0,0,0],[45,0,0]]
        q = self.p.query(pts)
        for k in range(4):
            self.assertEqual(q['m0'][k], self.p.tissues[k][2])
            if k: self.assertEqual((q['t1_ms'][k], q['t2_ms'][k]), self.p.tissues[k][:2])
        self.assertTrue(np.isnan(q['t1_ms'][0]) and np.isnan(q['t2_ms'][0]))
        np.testing.assert_array_equal(q['valid'], [False, True, True, True])
        tissues = self.p.tissues.copy(); tissues[1] = (999, 33, .8)
        np.testing.assert_array_equal(q['label'], Phantom(tissues=tissues, amplitude=0).query(pts)['label'])
        self.assertEqual(np.shape(self.p.query([0,0,0])['label']), ())
        self.assertEqual(self.p.query(np.zeros((2,3,3)))['label'].shape, (2,3))
    def test_demo(self):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([sys.executable, '-m', 'cardiac_phantom.demo', '--output-root', d], check=True, env={**os.environ,'MPLBACKEND':'Agg'})
            run = next(Path(d).iterdir())
            for f in ['anatomy.png','quantitative.png','sax_mid.npz','summary.json']: self.assertTrue((run/f).is_file())
            with np.load(run/'sax_mid.npz') as data: self.assertEqual(data['xyz_mm'].shape, (160,160,3))
            self.assertEqual(len(json.loads((run/'summary.json').read_text())['views']), 6)

if __name__ == '__main__': unittest.main()
