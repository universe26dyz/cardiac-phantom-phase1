"""Render 10-weight fitted signal comparisons for high-resolution MIND inputs."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/data2/dengyz/dataset/phantom/v6/l100_pilot_1p5mm_20261120T013453_650556Z')
out=root/'signal_comparison_mind_12000';out.mkdir(exist_ok=False)
summary={'geometry':{'pixel_spacing_mm':1.5,'slice_thickness_mm':8,'shape_rc':[256,256]},'source':'MIND-registered synthetic noisy inputs; final-model native-plane signal reprojection at optimized rigid poses; K=8 Gaussian PSF prediction','cases':[]}
for case,recon in [('static','proposed_static_12000_mind'),('respiratory','proposed_respiratory_12000_mind')]:
 for view,idx in [('sax',6),('2ch',7),('4ch',5)]:
  src=root/'prepared_mind_v3'/case/view/'qc_clean_and_mind_noisy.npz'
  predfile=root/recon/'evaluation/signal_psf_K8'/f'signal_reprojection_{view}.npz'
  with np.load(src) as d: clean=d['clean'][idx];noisy=d['noisy'][idx]
  with np.load(predfile) as d:
   rows=np.flatnonzero(d['group_idx']==idx);rows=rows[np.argsort(d['weight_idx'][rows])]
   np.testing.assert_array_equal(d['weight_idx'][rows],np.arange(10));observed=d['observed'][rows];pred=d['predicted'][rows];valid=d['masks'][rows];timing=d['timing9_ms'][rows]
   np.testing.assert_allclose(observed,noisy,atol=0,rtol=0)
  err=pred-noisy;err_gt=pred-clean
  assert np.isfinite(pred[valid]).all()
  vals=np.concatenate([clean.ravel(),noisy.ravel(),pred.ravel()]);scale=float(np.percentile(vals,99.8));emax=max(float(np.percentile(np.concatenate([abs(err[valid]),abs(err_gt[valid])]),99.8)),1e-9)
  fig,axes=plt.subplots(5,10,figsize=(22,11),constrained_layout=True)
  for h in range(10):
   for j,image in enumerate((clean[h],noisy[h],pred[h],np.where(valid[h],err[h],np.nan),np.where(valid[h],err_gt[h],np.nan))):
    axes[j,h].imshow(image,origin='lower',interpolation='nearest',cmap='gray' if j<3 else 'coolwarm',vmin=0 if j<3 else -emax,vmax=scale if j<3 else emax)
    axes[j,h].set_xticks([]);axes[j,h].set_yticks([])
   axes[0,h].set_title(f'HB{h+1}',fontsize=10)
  for j,title in enumerate(('Clean GT (MIND flow)','Noisy input (MIND)','Final prediction','Prediction − input','Prediction − clean')):axes[j,0].set_ylabel(title,fontsize=10)
  fig.suptitle(f'{case} | {view.upper()} | slice ID {idx} | 1.5 mm, 8 mm | final pose, Gaussian K=8 | [0,{scale:.4g}]',fontsize=13)
  fig.colorbar(axes[2,0].images[0],ax=list(axes[:3,:].flat),shrink=.65,label='relative signal')
  fig.colorbar(axes[3,0].images[0],ax=list(axes[3:,:].flat),shrink=.65,label='relative signal error')
  path=out/f'{case}_{view}_weights.png';fig.savefig(path,dpi=125);plt.close(fig)
  summary['cases'].append({'case':case,'view':view,'slice_id_zero_based':idx,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'reprojection_sha256':hashlib.sha256(predfile.read_bytes()).hexdigest(),'timing9_ms':timing[0].tolist(),'rmse_to_mind_noisy_inside_mask':float(np.sqrt(np.mean(err[valid]**2))),'rmse_to_mind_clean_inside_mask':float(np.sqrt(np.mean(err_gt[valid]**2))),'figure':str(path)})
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary['cases'],indent=2))
