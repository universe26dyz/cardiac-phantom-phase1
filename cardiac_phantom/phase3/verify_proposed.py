"""Read exported cases with the real Proposed loader and approved checkpoint."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from ismrm_v2.proposed.quantitative_point_dataset import QuantPointDataset
from ismrm_v2.proposed.signal_decoder.checkpoint_loader import load_frozen_mlp_decoder

def main():
 ap=argparse.ArgumentParser();ap.add_argument('handoff',type=Path);a=ap.parse_args(); names=('sax','2ch','4ch');paths=[a.handoff/(n+'_observations.npz') for n in names];q=QuantPointDataset(paths,'cpu');xyz=q.xyz_transformed.numpy(); records=json.loads((a.handoff/'geometry_and_sources.json').read_text()); results=[];offset=0
 for name,path in zip(names,paths):
  data=np.load(path);count=len(np.unique(data['group_idx']))
  for j in range(count):
   for hb in range(10):
    selected=(q.group_idx.numpy()==offset+j)&(q.weight_idx.numpy()==hb);np.testing.assert_array_equal(q.v.numpy()[selected],data['images'][j*10+hb].reshape(-1));np.testing.assert_allclose(q.timing.numpy()[selected][0],data['timing9_ms'][j*10+hb],atol=.0001)
    if hb==0:
     expected=np.array(next(x for x in records if x['stack']==name and x['slice_id']==j)['geometry_landmark_world']);np.testing.assert_allclose(xyz[selected][7*32+19],expected,atol=2e-5)
   results.append({'stack':name,'slice_id':j,'status':'pass','weights':list(range(10)),'pixel_values':'exact float32','geometry_landmark':'pass','timing':'pass','input_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
  offset+=count
 ck=Path('/data2/dengyz/code/ismrm/ismrm_v2_1/runtime/model_assets/decoder_scmr_v5_10_100.pth');model=load_frozen_mlp_decoder(ck,q,'/data2/dengyz/code/ismrm/ismrm_v2_3/configs/protocol_scmr_v5_10_100.yaml',allow_functional_fixture=False,device=torch.device('cpu'))
 with torch.no_grad(): prediction=model(torch.tensor([1200.,1600.]),torch.tensor([45.,180.]),torch.ones(2),q.timing[:2]); assert torch.isfinite(prediction).all()
 result={'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),'group_count':q.group_pose_count,'slices':results,'decoder_smoke':'pass','coverage':'per-dimension bounds; joint sample coverage not established'}
 out=a.handoff/'proposed_verification.json';
 if out.exists():raise FileExistsError(out)
 out.write_text(json.dumps(result,indent=2));print('Proposed validated',len(results),'slices')
if __name__=='__main__':main()
