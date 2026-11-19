"""Optional forward DVF on the canonical reference world grid."""
import argparse, json
from pathlib import Path
import numpy as np
import nibabel as nib
from .export import grid, xyz_grid
from .motion import Motion

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('output',type=Path); ap.add_argument('--time',type=float,required=True); ap.add_argument('--spacing',type=float,default=2.); a=ap.parse_args()
    if a.output.exists() or a.output.with_suffix(a.output.suffix+'.json').exists(): raise FileExistsError(a.output)
    shape,aff=grid(a.spacing); m=Motion(); xyz=xyz_grid(shape,aff); vectors=m.dvf(xyz,a.time).astype(np.float32)[...,None,:]
    img=nib.Nifti1Image(vectors,aff); img.set_qform(aff,1); img.set_sform(aff,1); img.header.set_xyzt_units('mm'); img.header.set_intent('vector'); nib.save(img,a.output)
    a.output.with_suffix(a.output.suffix+'.json').write_text(json.dumps({'direction':'forward','grid':'reference world','vector_basis':'world xyz','unit':'mm','axes':'X,Y,Z,singleton,vector','time_s':a.time,'affine':aff.tolist()},indent=2))
if __name__=='__main__': main()
