"""Pack high-res continuous-phantom inputs for MATLAB MIND and Q002."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage import label

def pack_main(run_root):
 root=Path(run_root); io=root/'matlab_io';io.mkdir(exist_ok=False)
 manifest=json.loads((root/'manifest.json').read_text())
 report={'run_root':str(root),'mind_method':'ismrm_v2_3 MATLAB preprocessing_v1 deformableReg2Dmind_asym_nodisplay; HB1 reference; alpha=0.3','clean_transform':'same flow estimated from noisy input','groups':{}}
 for case in ('static','respiratory'):
  for view,count in [('sax',12),('2ch',15),('4ch',10)]:
   src=np.load(root/case/(view+'.npz'))
   noisy=np.transpose(src['noisy'],(2,3,1,0)).astype('f4',copy=False)
   clean=np.transpose(src['clean'],(2,3,1,0)).astype('f4',copy=False)
   out=io/f'{case}_{view}_mind_input.mat'
   savemat(out,{'noisy':noisy,'clean':clean,'case_name':case,'view_name':view},do_compression=True,format='5')
   report['groups'][f'{case}/{view}']={'mat_input':str(out),'shape_row_col_hb_slice':list(noisy.shape),'slice_count':count,'weights':10}
 (io/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 print(io)

def prepare_observations(run_root, reference_prepared, registered_root, output_root):
 run_root=Path(run_root);reference_prepared=Path(reference_prepared);registered_root=Path(registered_root);output_root=Path(output_root)
 manifest=json.loads((run_root/'manifest.json').read_text())
 output_root.mkdir(parents=True,exist_ok=False)
 qcreport={'source_run':str(run_root),'mind_root':str(registered_root),'input_psf':manifest['sampling']['input_psf'],'pixel_spacing_mm':1.5,'slice_thickness_mm':8,'mask_rule':'per-slice 10-weight mean > border median + 4 * (1.4826 * border MAD); retain 2D components >=20 pixels; border width=16 pixels','cases':{}}
 for case in ('static','respiratory'):
  (output_root/case).mkdir()
  for view,count in [('sax',12),('2ch',15),('4ch',10)]:
   x=loadmat(registered_root/f'{case}_{view}_mind_output.mat',variable_names=['registered_noisy','registered_clean','registration_qc'])
   noisy=np.transpose(x['registered_noisy'],(3,2,0,1)).astype('f4',copy=False) # slice,HB,row,col
   clean=np.transpose(x['registered_clean'],(3,2,0,1)).astype('f4',copy=False)
   refdata=np.load(reference_prepared/view/'observations.npz')
   # Build group masks from the actual registered noisy input.
   images=noisy.reshape(count*10,*noisy.shape[-2:]); group_means=noisy.mean(axis=1); group_masks=[]; noise_sigmas=[]; component_counts=[]
   for g in group_means:
    border=np.concatenate((g[:16].ravel(),g[-16:].ravel(),g[:,:16].ravel(),g[:,-16:].ravel())); med=float(np.median(border)); sigma=float(1.4826*np.median(np.abs(border-med))); candidate=np.isfinite(g)&(g>med+4*sigma); labels,nlabels=label(candidate); sizes=np.bincount(labels.ravel()); keep=np.flatnonzero(sizes[1:]>=20)+1; foreground=np.isin(labels,keep);
    if not foreground.any(): raise ValueError(f'empty noise-calibrated mask: {case}/{view}')
    group_masks.append(foreground);noise_sigmas.append(sigma);component_counts.append(int(len(keep)))
   group_masks=np.stack(group_masks); masks=np.repeat(group_masks[:,None],10,axis=1).reshape(count*10,*noisy.shape[-2:])
   if not masks.any(): raise ValueError(f'empty masks: {case}/{view}')
   old_aff=refdata['affine_lps_rc'].reshape(count,10,4,4)[:,0].copy(); aff=old_aff.copy()
   old_center=old_aff[:,:3,3]+old_aff[:,:3,0]*15.5+old_aff[:,:3,1]*15.5
   aff[:,:3,0]=old_aff[:,:3,0]*(1.5/5.0);aff[:,:3,1]=old_aff[:,:3,1]*(1.5/5.0)
   aff[:,:3,3]=old_center-aff[:,:3,0]*127.5-aff[:,:3,1]*127.5
   aff_by_obs=np.repeat(aff,10,axis=0)
   timing=np.asarray([np.asarray(row['timing9_ms']) for row in manifest['slices'] if row['stack']==view],dtype=np.float64)
   if timing.shape!=(count,9): raise ValueError(f'timing shape {timing.shape}: {case}/{view}')
   acq=np.asarray([row['acquisition_time_ms'] for row in manifest['slices'] if row['stack']==view],dtype=np.float64).reshape(-1)
   out=output_root/case/view;out.mkdir(exist_ok=True)
   np.savez_compressed(out/'observations.npz',images=images,masks=masks,group_idx=np.repeat(np.arange(count),10),weight_idx=np.tile(np.arange(10),count),stack_idx=np.full(count*10,{'sax':0,'2ch':1,'4ch':2}[view]),acquisition_time_ms=acq,timing9_ms=np.repeat(timing,10,axis=0),affine_lps_rc=aff_by_obs,pixel_spacing_rc_mm=np.full((count*10,2),1.5),slice_thickness_mm=np.full(count*10,8.0),tr_ms=np.full(count*10,2.61),vps=np.full(count*10,87,dtype=np.int64))
   np.savez_compressed(out/'qc_clean_and_mind_noisy.npz',clean=clean,noisy=noisy,noise=np.transpose(np.load(run_root/case/(view+'.npz'))['noise'],(0,1,2,3)))
   registration_qc=np.asarray(x['registration_qc']); qcreport['cases'][f'{case}/{view}']={'shape_slice_hb_row_col':list(noisy.shape),'mask_pixels_per_group':masks.reshape(count,10,-1)[:,0].sum(axis=1).tolist(),'timing_groups':count,'source_group_ids':list(range(count)),'mask_noise_sigma_median':float(np.median(noise_sigmas)),'mask_component_count_max':int(max(component_counts)),'mind_registration_qc':{'pre_mse_mean':float(registration_qc[:,1:,0].mean()),'post_mse_mean':float(registration_qc[:,1:,1].mean()),'median_displacement_pixels_median':float(np.median(registration_qc[:,1:,2])),'max_displacement_pixels':float(registration_qc[:,:,3].max())}}
 (output_root/'preparation_qc.json').write_text(json.dumps(qcreport,indent=2)+'\n')
 print(output_root)

def main():
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='action',required=True)
 p=sub.add_parser('pack');p.add_argument('run_root',type=Path)
 p=sub.add_parser('prepare');p.add_argument('run_root',type=Path);p.add_argument('reference_prepared',type=Path);p.add_argument('registered_root',type=Path);p.add_argument('output_root',type=Path)
 a=ap.parse_args()
 if a.action=='pack': pack_main(a.run_root)
 else: prepare_observations(a.run_root,a.reference_prepared,a.registered_root,a.output_root)

if __name__=='__main__': main()
