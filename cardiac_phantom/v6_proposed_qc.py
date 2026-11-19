"""Evaluate available Proposed maps on the synthetic reference grid."""
import argparse,json
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cardiac_phantom.v6_evaluate import sample

def main():
 ap=argparse.ArgumentParser();ap.add_argument('reconstruction',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 root=Path('/data2/dengyz/dataset/phantom/phase1_fix/fix_20261119T072307_890862Z/static_3d'); label=nib.load(root/'label.nii.gz');lab=np.asarray(label.dataobj);shape=lab.shape;world=nib.affines.apply_affine(label.affine,np.indices(shape).reshape(3,-1).T);support=sample(a.reconstruction/'support_mask_3D.nii.gz',world,shape,order=0)>.5;maps={k:sample(a.reconstruction/(k.upper()+'_3D.nii.gz'),world,shape) for k in ('t1','t2')};truth={k:np.asarray(nib.load(root/(k+'_ms.nii.gz')).dataobj) for k in maps}
 for v in maps.values():support &= np.isfinite(v)&(v>0)
 nib.save(nib.Nifti1Image(support.astype('u1'),label.affine),a.output/'proposed_support.nii.gz');rows={}
 for k,name in [(1,'lv_myocardium'),(2,'lv_blood'),(3,'rv_blood')]:
  roi=lab==k;sel=roi&support;n=int(sel.sum());rows[name]={'total':int(roi.sum()),'supported':n,'coverage':n/int(roi.sum())}
  for x in maps:
   err=(maps[x]-truth[x])[sel];rows[name][x]={'rmse_ms':float(np.sqrt(np.mean(err**2))) if n else None,'mae_ms':float(np.mean(abs(err))) if n else None}
 counts=(support & (lab>0)).sum(axis=(0,1)); z=int(np.argmax(counts)); print('QC z index',z,'supported tissue voxels',int(counts[z]))
 for name,limit in [('t1',1800),('t2',200)]:
  fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
  for i,(arr,title) in enumerate(((truth[name],'truth: tissue'),(maps[name],'proposed: tissue region'),(maps[name]-truth[name],'error: support only'))):
   display_mask=(lab>0) if i<2 else (support & (lab>0))
   im=axes[i].imshow(np.where(display_mask,arr,np.nan)[:,:,z].T,origin='lower',cmap='viridis' if i<2 else 'coolwarm',vmin=0 if i<2 else -limit/4,vmax=limit if i<2 else limit/4);axes[i].set_title(title);fig.colorbar(im,ax=axes[i])
  fig.savefig(a.output/(name+'_qc.png'),dpi=150);plt.close(fig)
 (a.output/'metrics.json').write_text(json.dumps({'note':'Proposed-only observed pixel-center support; not common-support comparison with Baseline','qc_slice_z_index':z,'regions':rows},indent=2));print(json.dumps(rows))
if __name__=='__main__':main()
