"""One fixed SAX slice: per-HB moving tissue signal, through-plane box PSF, then noise."""
import argparse, hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from cardiac_phantom.motion import Motion
from cardiac_phantom.phase3.signal import fingerprints, timing9, SOURCE, SOURCE_SHA256

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-root',default='/data2/dengyz/dataset/phantom/phase3'); ap.add_argument('--sigma',type=float,default=.002); ap.add_argument('--seed',type=int,default=123); a=ap.parse_args()
    if a.sigma < 0: raise ValueError('sigma must be nonnegative')
    out=Path(a.output_root)/datetime.now(timezone.utc).strftime('smoke_%Y%m%dT%H%M%S_%fZ'); out.mkdir(parents=True,exist_ok=False)
    motion=Motion(); n=32; spacing=3.; rr=1030.; times=np.arange(10)*rr; timings=timing9(times)
    y,x=np.meshgrid((np.arange(n)-(n-1)/2)*spacing,(np.arange(n)-(n-1)/2)*spacing,indexing='ij')
    offsets=np.array([-8/3,0.,8/3]); subweights=np.full(3,1/3)
    coords=np.stack([np.broadcast_to(x,(3,n,n)),np.broadcast_to(y,(3,n,n)),np.broadcast_to(12.37+offsets[:,None,None],(3,n,n))],axis=-1)
    clean=np.zeros((10,n,n),np.float32); support=np.zeros((n,n),bool)
    for k,t in enumerate(times):
        q=motion.query(coords,t/1000); valid=q['valid']; support|=valid.any(axis=0)
        response=np.zeros(valid.shape,np.float32)
        if valid.any(): response[valid]=fingerprints(q['t1_ms'][valid],q['t2_ms'][valid],timings)[:,k]*q['m0'][valid]
        clean[k]=np.tensordot(subweights,response,axes=(0,0))
    noise=np.random.default_rng(a.seed).normal(0,a.sigma,clean.shape).astype(np.float32); noisy=clean+noise
    np.savez_compressed(out/'observations.npz',clean=clean,noisy=noisy,mask=support,acquisition_time_ms=times,timing9_ms=timings,center_mm=[0,0,12.37],row_unit=[0,1,0],col_unit=[1,0,0],normal_unit=[0,0,1],spacing_mm=spacing,psf_offsets_mm=offsets,psf_weights=subweights)
    artifact=out/'observations.npz'; info={'status':'smoke; no verified downstream adapters','git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'working_tree_status':subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'reference_simulator':str(SOURCE),'reference_sha256':SOURCE_SHA256,'protocol':{'tr_ms':2.61,'vps':87,'ramp_up':10,'TI_ms':[10,100],'T2prep_ms':[35,45,55],'B1':1},'timing9_ms':timings.tolist(),'acquisition_time_ms':times.tolist(),'psf':'box width 8 mm; 3 midpoint samples; no in-plane integration','noise':{'seed':a.seed,'sigma':a.sigma,'residual_mean':float(noise.mean()),'residual_std':float(noise.std())},'sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'limitations':['image-domain real magnitude mixture','one representative time per heartbeat','MATLAB parity verified for one reference point only','synthetic geometry','no downstream handoff verified']}
    (out/'summary.json').write_text(json.dumps(info,indent=2)+'\n'); print(out)
if __name__=='__main__': main()
