"""NIfTI samples use voxel-center world coordinates in (X,Y,Z) order."""
from pathlib import Path
import numpy as np
import nibabel as nib

KEYS = ('label','t1_ms','t2_ms','m0','valid')
def grid(spacing=1.):
    shape = tuple(int(v/spacing)+1 for v in (120, 100, 140))
    affine = np.diag([spacing,spacing,spacing,1.]); affine[:3,3] = [-50,-50,-70]
    return shape, affine

def xyz_grid(shape, affine):
    idx = np.indices(shape, dtype=np.float32).reshape(3,-1).T
    return nib.affines.apply_affine(affine, idx).reshape((*shape,3))

def sample_grid(query, shape, affine):
    out = {k: np.empty(shape, dtype=np.uint8 if k in ('label','valid') else np.float32) for k in KEYS}
    for z in range(shape[2]):
        xyz = xyz_grid((shape[0],shape[1],1), affine); xyz[...,2] += z*affine[2,2]
        q = query(xyz[:,:,0,:])
        for k in KEYS: out[k][:,:,z] = q[k]
    return out

def save_maps(directory, maps, affine, dt=None):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    for k, array in maps.items():
        img = nib.Nifti1Image(array, affine); img.set_qform(affine, code=1); img.set_sform(affine, code=1)
        img.header.set_xyzt_units('mm', 'sec')
        if dt is not None: img.header.set_zooms((*img.header.get_zooms()[:3],dt))
        nib.save(img, directory/(k+'.nii.gz'))
