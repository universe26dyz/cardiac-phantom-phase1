"""256x256, 1.5 mm in-plane paired L100 continuous-phantom engineering pilot."""
import argparse, json, subprocess
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from cardiac_phantom.motion import Motion
from cardiac_phantom.phase3.signal import fingerprints, timing9
from cardiac_phantom.phase3.psf import integrate

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-root',default='/data2/dengyz/dataset/phantom/v6')
    ap.add_argument('--seed',type=int,default=123)
    ap.add_argument('--sigma',type=float,default=.002)
    ap.add_argument('--size',type=int,default=256)
    ap.add_argument('--spacing-mm',type=float,default=1.5)
    a=ap.parse_args()
    if a.size != 256 or a.spacing_mm != 1.5:
        raise ValueError('This experiment is frozen at 256x256 and 1.5 mm in-plane')
    out=Path(a.output_root)/datetime.now(timezone.utc).strftime('l100_pilot_1p5mm_%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True,exist_ok=False)
    for kind in ('static','respiratory'): (out/kind).mkdir()
    motion=Motion(); rng=np.random.default_rng(a.seed); n=a.size; spacing=a.spacing_mm; rr=1000.
    offsets=np.array([-8/3,0,8/3]); psf_weights=np.full(3,1/3)
    grid=(np.arange(n)-(n-1)/2)*spacing; r,c=np.meshgrid(grid,grid,indexing='ij')
    meta=[]
    definitions=[('sax',12,[0,0,1],[0,1,0],[1,0,0],np.linspace(-40,40,12)),('2ch',15,[1,0,0],[0,0,1],[0,1,0],np.linspace(-20,20,15)),('4ch',10,[0,1,0],[0,0,1],[1,0,0],np.linspace(-18,18,10))]
    for si,(name,count,normal,row,col,positions) in enumerate(definitions):
        normal,row,col=map(lambda v:np.asarray(v,float),(normal,row,col))
        clean={k:[] for k in ('static','respiratory')}; noisy={k:[] for k in clean}; noise_list=[]
        for j,position in enumerate(positions):
            center=normal*position
            xyz=center[None,None,None,:]+r[None,:,:,None]*row[None,None,None,:]+c[None,:,:,None]*col[None,None,None,:]+offsets[:,None,None,None]*normal[None,None,None,:]
            times=(si*20+j)*10300+np.arange(10)*rr
            timing=timing9(times)
            obs={k:np.zeros((10,n,n),np.float32) for k in clean}
            for hb,t in enumerate(times):
                for kind in clean:
                    q=motion.query(xyz,t/1000,cardiac=False,respiratory=(kind=='respiratory'))
                    valid=q['valid']; signal=np.zeros(valid.shape)
                    if valid.any(): signal[valid]=fingerprints(q['t1_ms'][valid],q['t2_ms'][valid],timing)[:,hb]*q['m0'][valid]
                    obs[kind][hb]=integrate(signal,psf_weights)
            noise=rng.normal(0,a.sigma,(10,n,n)).astype('f4'); noise_list.append(noise)
            for kind in clean:
                clean[kind].append(obs[kind]); noisy[kind].append((obs[kind]+noise).astype('f4'))
            meta.append({'stack':name,'slice_id':j,'center_mm':center.tolist(),'row_unit':row.tolist(),'column_unit':col.tolist(),'normal_unit':normal.tolist(),'shape_rc':[n,n],'spacing_rc_mm':[spacing,spacing],'slice_thickness_mm':8,'psf_offsets_mm':offsets.tolist(),'psf_weights':psf_weights.tolist(),'acquisition_time_ms':times.tolist(),'timing9_ms':timing.tolist(),'cardiac_phase_cycles':[0.0]*10,'respiratory_phase_cycles':((times/1000)%motion.respiratory_period/motion.respiratory_period).tolist()})
            print(name,j+1,'/',count,flush=True)
        for kind in clean:
            np.savez_compressed(out/kind/(name+'.npz'),clean=np.asarray(clean[kind]),noisy=np.asarray(noisy[kind]),noise=np.asarray(noise_list))
    manifest={'label':'L100 256x256 1.5mm in-plane, 8mm slice engineering pilot; not formal performance','git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'dirty':subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'array_order':'slice,heartbeat,row,column','noise':{'type':'paired additive Gaussian after PSF','seed':a.seed,'sigma':a.sigma},'motion':{'cardiac':'disabled; ED phase 0','respiratory_period_s':motion.respiratory_period,'ap_mm':motion.respiratory_ap_mm,'si_mm':motion.respiratory_si_mm},'protocol':{'ti_ms':[10,100],'tr_ms':2.61,'vps':87,'rr_ms':rr},'sampling':{'shape_rc':[n,n],'spacing_rc_mm':[spacing,spacing],'fov_rc_mm':[n*spacing,n*spacing],'slice_thickness_mm':8,'input_psf':'three-point box integral along 8mm slice thickness'},'slices':meta}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('OUTPUT',out,flush=True)
if __name__=='__main__':main()
