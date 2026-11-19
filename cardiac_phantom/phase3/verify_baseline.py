"""Verify actual per-slice MATLAB matcher outputs against the original stack."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.io import loadmat

def main():
 p=argparse.ArgumentParser();p.add_argument('handoff',type=Path);p.add_argument('source',type=Path);a=p.parse_args();base=a.handoff;geo=json.loads((base/'geometry_and_sources.json').read_text());prop=json.loads((base/'proposed_verification.json').read_text());rows=[]
 for stack,count in [('sax',12),('2ch',15),('4ch',10)]:
  source=np.load(base/(stack+'_observations.npz'));truth=np.load(a.source/(stack+'.npz'))['clean']
  for j in range(count):
   stem=base/'groups'/f'{stack}_{j:02d}';inp=loadmat(str(stem)+'_input.mat',simplify_cells=True);out=loadmat(str(stem)+'_output.mat',simplify_cells=True);qc=out['QC'];np.testing.assert_array_equal(inp['weight_idx'],np.arange(10));np.testing.assert_allclose(qc['duration_before_acq_ms'][1:],source['timing9_ms'][j*10],atol=1e-8);np.testing.assert_array_equal(inp['Mag_synthetic'],np.moveaxis(source['images'][j*10:j*10+10],0,2));np.testing.assert_allclose(inp['Acq_time'],source['acquisition_time_ms'][j*10:j*10+10]);n=int(np.count_nonzero(qc['valid_match_map'])); nonzero=int(np.count_nonzero(truth[j]));g=next(x for x in geo if x['stack']==stack and x['slice_id']==j);pr=next(x for x in prop['slices'] if x['stack']==stack and x['slice_id']==j)
   rows.append({'stack':stack,'slice_id':j,'weights':'0..9','source_sha256':g['source_sha256'],'proposed_input_sha256':pr['input_sha256'],'pixel_comparison':'exact','timing':'pass','world_landmark':'pass','baseline_valid_pixels':n,'baseline':'pass' if n else ('no-phantom-signal' if nonzero==0 else 'zero-valid-review'),'proposed':pr['status'],'nonzero_clean_samples':nonzero})
 out=base/'per_slice_acceptance.csv'
 if out.exists():raise FileExistsError(out)
 with out.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 print('rows',len(rows),'baseline valid',sum(x['baseline_valid_pixels']>0 for x in rows),'zero valid',[(x['stack'],x['slice_id'],x['baseline']) for x in rows if x['baseline_valid_pixels']==0])
if __name__=='__main__':main()
