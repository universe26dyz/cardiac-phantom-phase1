"""Export existing immutable Phase 3 observations to both downstream formats."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.io import savemat

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 (a.output/'groups').mkdir(); manifest=json.loads((a.source/'manifest.json').read_text()); transform=np.diag([-1.,-1.,1.]); report=[]
 for idx,(name,count) in enumerate((('sax',12),('2ch',15),('4ch',10))):
  path=a.source/(name+'.npz');d=np.load(path); images=d['noisy']; assert images.shape==(count,10,32,32); items=[s for s in manifest['slices'] if s['stack']==name]; assert len(items)==count
  aff=[]; times=[]; timings=[]
  for j,s in enumerate(items):
   assert s['slice_id']==j;row=np.asarray(s['row_unit']);col=np.asarray(s['column_unit']);n=np.cross(col,row);center=np.asarray(s['center_mm']);sp=s['spacing_mm']; A=np.eye(4);A[:3,0]=transform@(row*sp);A[:3,1]=transform@(col*sp);A[:3,2]=transform@(n*s['slice_thickness_mm']);A[:3,3]=transform@(center-15.5*sp*(row+col));
   # A is LPS row/column; prove an asymmetric pixel and its inverse.
   rc=np.array([7.,19.,0.,1.]);expected=center+(7-15.5)*sp*row+(19-15.5)*sp*col;np.testing.assert_allclose(transform@(A@rc)[:3],expected,atol=1e-10);np.testing.assert_allclose(np.linalg.inv(A)@np.r_[transform@expected,1.],rc,atol=1e-10)
   aff.append(A);times.append(s['acquisition_time_ms']);timings.append(s['timing9_ms']);report.append({'stack':name,'slice_id':j,'source_sha256':sha(path),'normal_sample':s['normal_unit'],'normal_affine':n.tolist(),'geometry_landmark_rc':[7,19],'geometry_landmark_world':expected.tolist()})
  x=images.reshape(count*10,32,32);mask=np.ones_like(x,dtype=bool);t=np.array(times);q=np.array(timings);p=a.output/(name+'_observations.npz'); np.savez_compressed(p,images=x,masks=mask,group_idx=np.repeat(np.arange(count),10),weight_idx=np.tile(np.arange(10),count),stack_idx=np.full(count*10,idx),acquisition_time_ms=t.reshape(-1),timing9_ms=np.repeat(q,10,axis=0),affine_lps_rc=np.repeat(aff,10,axis=0),pixel_spacing_rc_mm=np.full((count*10,2),5.),slice_thickness_mm=np.full(count*10,8.),tr_ms=np.full(count*10,2.61),vps=np.full(count*10,87))
  mat=a.output/(name+'_matcher_input.mat');savemat(mat,{'Mag_synthetic':np.transpose(images,(2,3,1,0)),'Acq_time':t.T,'group_idx':np.arange(count),'weight_idx':np.arange(10),'source_sha256':sha(path)});from scipy.io import loadmat
  for j in range(count):
   savemat(a.output/'groups'/f'{name}_{j:02d}_input.mat',{'Mag_synthetic':np.moveaxis(images[j],0,2)[:,:,:,None],'Acq_time':t[j,:,None],'group_idx':np.array([0]),'weight_idx':np.arange(10),'source_sha256':sha(path)})
  m=loadmat(mat);np.testing.assert_array_equal(np.transpose(m['Mag_synthetic'],(3,2,0,1)),images);np.testing.assert_allclose(m['Acq_time'],t.T)
  with np.load(p) as z: np.testing.assert_array_equal(z['images'],x)
  print(name,'converted',sha(path),sha(p),sha(mat),flush=True)
 (a.output/'geometry_and_sources.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
