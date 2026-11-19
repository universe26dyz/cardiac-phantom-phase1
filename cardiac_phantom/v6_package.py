"""Package actual SCMR-v5 MATLAB map outputs for the frozen NeSVoR exporter."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from ismrm_v2.fit_first.fit_first import export_nesvor_native_inputs

def main():
 ap=argparse.ArgumentParser();ap.add_argument('pilot',type=Path);ap.add_argument('group',choices=('static','respiratory'));a=ap.parse_args();root=a.pilot; source=root/('handoff_'+a.group);target=root/('baseline_inputs_'+a.group);target.mkdir(exist_ok=False);report={}
 for name,count in [('sax',12),('2ch',15),('4ch',10)]:
  obs=np.load(root/('prepared_'+a.group)/name/'observations.npz');t1=[];t2=[];valid_counts=[]
  for i in range(count):
   stem=source/'groups'/f'{name}_{i:02d}';data=loadmat(str(stem)+'_input.mat',simplify_cells=True); result=loadmat(str(stem)+'_output.mat',simplify_cells=True); qc=result['QC'];np.testing.assert_array_equal(data['Mag_synthetic'],np.moveaxis(obs['images'][i*10:i*10+10],0,2));np.testing.assert_allclose(qc['duration_before_acq_ms'][1:],obs['timing9_ms'][i*10]);mask=qc['valid_match_map'].astype(bool);valid_counts.append(int(mask.sum()));t1.append(np.where(mask,result['T1'],np.nan).astype('f4'));t2.append(np.where(mask,result['T2'],np.nan).astype('f4'))
  folder=target/(name+'_maps');folder.mkdir();np.savez_compressed(folder/'native_maps.npz',t1_ms=np.array(t1),t2_ms=np.array(t2),group_idx=np.arange(count));np.savez_compressed(folder/'geometry.npz',affine_lps_rc=obs['affine_lps_rc'][::10],pixel_spacing_rc_mm=obs['pixel_spacing_rc_mm'][::10],slice_thickness_mm=obs['slice_thickness_mm'][::10]);export_nesvor_native_inputs(folder,target/name,normalization={'t1':2500.,'t2':200.});report[name]={'valid_pixels':valid_counts,'input_sha256':hashlib.sha256((root/('prepared_'+a.group)/name/'observations.npz').read_bytes()).hexdigest()}
 (target/'package_provenance.json').write_text(json.dumps(report,indent=2));print(target)
if __name__=='__main__':main()
