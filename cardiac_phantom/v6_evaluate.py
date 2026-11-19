"""Compare maps in physical space on the fixed synthetic reference grid."""
import argparse,json
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sample(path,world,shape):
 im=nib.load(path); data=np.asanyarray(im.dataobj); vox=nib.affines.apply_affine(np.linalg.inv(im.affine),world); return map_coordinates(data,vox.T,order=1,mode='constant',cval=np.nan).reshape(shape)

def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline-t1',required=True);p.add_argument('--baseline-t2',required=True);p.add_argument('--proposed-t1',required=True);p.add_argument('--proposed-t2',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 ref=Path('/data2/dengyz/dataset/phantom/phase1_fix/fix_20261119T072307_890862Z/static_3d');labels=nib.load(ref/'label.nii.gz');lab=np.asanyarray(labels.dataobj);shape=lab.shape; xyz=np.indices(shape).reshape(3,-1).T;world=nib.affines.apply_affine(labels.affine,xyz)
 gt={k:np.asanyarray(nib.load(ref/(k+'_ms.nii.gz')).dataobj) for k in ('t1','t2')};maps={'baseline':{'t1':sample(a.baseline_t1,world,shape),'t2':sample(a.baseline_t2,world,shape)},'proposed':{'t1':sample(a.proposed_t1,world,shape),'t2':sample(a.proposed_t2,world,shape)}}
 support=np.ones(shape,bool)
 for model in maps.values():
  for x in model.values():support &= np.isfinite(x)&(x>0)
 nib.save(nib.Nifti1Image(support.astype('u1'),labels.affine),a.output/'common_support.nii.gz');stats={}
 for k,name in [(1,'lv_myocardium'),(2,'lv_blood'),(3,'rv_blood')]:
  tissue=(lab==k);sel=tissue&support; n=int(sel.sum());stats[name]={'voxels':int(tissue.sum()),'common_voxels':n,'coverage':n/int(tissue.sum()),'methods':{}}
  for model in maps:
   stats[name]['methods'][model]={}
   for param in ('t1','t2'):
    e=(maps[model][param]-gt[param])[sel];stats[name]['methods'][model][param]={'rmse_ms':float(np.sqrt(np.mean(e**2))) if n else None,'mae_ms':float(np.mean(np.abs(e))) if n else None}
 z=shape[2]//2
 for param,lim in [('t1',1800),('t2',200)]:
  fig,ax=plt.subplots(1,5,figsize=(16,4),constrained_layout=True); images=[gt[param],maps['baseline'][param],maps['proposed'][param],maps['baseline'][param]-gt[param],maps['proposed'][param]-gt[param]]
  for i,(a1,im,title) in enumerate(zip(ax,images,('truth','baseline','proposed','baseline error','proposed error'))):
   obj=a1.imshow(np.where(support,im,np.nan)[:,:,z].T,origin='lower',cmap='viridis' if i<3 else 'coolwarm',vmin=0 if i<3 else -lim/4,vmax=lim if i<3 else lim/4);a1.set_title(title);fig.colorbar(obj,ax=a1)
  fig.savefig(a.output/(param+'_comparison.png'),dpi=150);plt.close(fig)
 (a.output/'metrics.json').write_text(json.dumps({'reference':str(ref),'support':'finite positive values in both maps and both parameters on reference grid','regions':stats},indent=2)); print(a.output)
if __name__=='__main__':main()
