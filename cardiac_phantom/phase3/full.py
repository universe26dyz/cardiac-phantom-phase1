"""Generate a 37-slice continuous phantom observation example."""
import argparse, hashlib, json, subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cardiac_phantom.motion import Motion
from cardiac_phantom.phase3.psf import integrate,add_noise
from cardiac_phantom.phase3.signal import fingerprints,timing9

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output-root',default='/data2/dengyz/dataset/phantom/phase3'); parser.add_argument('--seed',type=int,default=123); parser.add_argument('--sigma',type=float,default=.002); args=parser.parse_args()
    out=Path(args.output_root)/datetime.now(timezone.utc).strftime('full_%Y%m%dT%H%M%S_%fZ'); out.mkdir(parents=True,exist_ok=False)
    m=Motion(); n=32; spacing=4.; rr=1030.; offsets=np.array([-8/3,0,8/3]); w=np.full(3,1/3); grid=(np.arange(n)-(n-1)/2)*spacing; r,c=np.meshgrid(grid,grid,indexing='ij'); rng=np.random.default_rng(args.seed)
    definitions=[('sax',12,[0,0,1],[0,1,0],[1,0,0],np.linspace(-40,40,12)),('2ch',15,[1,0,0],[0,0,1],[0,1,0],np.linspace(-20,20,15)),('4ch',10,[0,1,0],[0,0,1],[1,0,0],np.linspace(-18,18,10))]
    meta=[]; noise_all=[]
    for si,(name,count,normal,row,col,positions) in enumerate(definitions):
        normal,row,col=map(lambda x:np.array(x,dtype=float),(normal,row,col)); allclean=[]; allstatic=[]; allnoisy=[]
        for j,position in enumerate(positions):
            center=normal*position; xyz=center+r[None,:,:,None]*row+c[None,:,:,None]*col+offsets[:,None,None,None]*normal; times=(si*20+j)*12000+np.arange(10)*rr; timing=timing9(times)
            clean=np.zeros((10,n,n),np.float32); static=clean.copy()
            for k,t in enumerate(times):
                for dynamic,dest in [(True,clean),(False,static)]:
                    q=m.query(xyz,t/1000,cardiac=dynamic,respiratory=dynamic); valid=q['valid']; response=np.zeros(valid.shape)
                    if valid.any(): response[valid]=fingerprints(q['t1_ms'][valid],q['t2_ms'][valid],timing)[:,k]*q['m0'][valid]
                    dest[k]=integrate(response,w)
            noise=rng.normal(0,args.sigma,clean.shape); noisy=clean+noise; noise_all.append(noise.ravel()); allclean.append(clean); allstatic.append(static); allnoisy.append(noisy.astype('f4'))
            meta.append({'stack':name,'slice_id':j,'center_mm':center.tolist(),'row_unit':row.tolist(),'column_unit':col.tolist(),'normal_unit':normal.tolist(),'spacing_mm':spacing,'slice_thickness_mm':8,'psf_offsets_mm':offsets.tolist(),'psf_weights':w.tolist(),'acquisition_time_ms':times.tolist(),'timing9_ms':timing.tolist()}); print(name,j+1,'/',count,flush=True)
        np.savez_compressed(out/(name+'.npz'),clean=np.array(allclean),noisy=np.array(allnoisy),static=np.array(allstatic))
        fig,axes=plt.subplots(2,5,figsize=(12,5)); v=np.max(allclean[count//2]);
        for k,ax in enumerate(axes.flat): ax.imshow(allclean[count//2][k],origin='lower',vmin=0,vmax=v,cmap='gray');ax.set_title(f'HB{k+1}');ax.axis('off')
        fig.suptitle(name+' clean midpoint');fig.tight_layout();fig.savefig(out/(name+'_montage.png'));plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(12,4)); im=allclean[count//2][4]; axes[0].imshow(allstatic[count//2][4],origin='lower',cmap='gray',vmin=0,vmax=v);axes[0].set_title('static');axes[1].imshow(im,origin='lower',cmap='gray',vmin=0,vmax=v);axes[1].set_title('motion');axes[2].imshow(allnoisy[count//2][4]-im,origin='lower',cmap='coolwarm',vmin=-.006,vmax=.006);axes[2].set_title('noise residual');fig.savefig(out/(name+'_comparison.png'));plt.close(fig)
    # Compare three-point PSF with the zero-thickness center at one SAX event.
    view=meta[6]; center=np.asarray(view['center_mm']); xyz=center+r[...,None]*np.asarray(view['row_unit'])+c[...,None]*np.asarray(view['column_unit']); t=view['acquisition_time_ms'][4]; timing=np.asarray(view['timing9_ms']); q=m.query(xyz,t/1000); mask=q['valid']; point=np.zeros((n,n)); point[mask]=fingerprints(q['t1_ms'][mask],q['t2_ms'][mask],timing)[:,4]*q['m0'][mask]; thick=np.load(out/'sax.npz')['clean'][6,4]; fig,ax=plt.subplots(1,3,figsize=(12,4)); ax[0].imshow(point,origin='lower',cmap='gray');ax[0].set_title('zero thickness');ax[1].imshow(thick,origin='lower',cmap='gray');ax[1].set_title('8 mm box PSF');im=ax[2].imshow(thick-point,origin='lower',cmap='coolwarm');ax[2].set_title('PSF difference');fig.colorbar(im,ax=ax[2]);fig.savefig(out/'psf_comparison.png');plt.close(fig)
    checkpoint=Path('/data2/dengyz/code/ismrm/ismrm_v2_1/runtime/model_assets/decoder_scmr_v5_10_100.pth'); ck=torch.load(checkpoint,map_location='cpu',weights_only=False); lo=np.array(ck['train_timing9_min_ms']);hi=np.array(ck['train_timing9_max_ms']); outliers=[{'stack':s['stack'],'slice_id':s['slice_id']} for s in meta if np.any((np.array(s['timing9_ms'])<lo)|(np.array(s['timing9_ms'])>hi))]
    noise=np.concatenate(noise_all); manifest={'status':'synthetic illustrative acquisition','commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'working_tree':subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'shape':'slice,heartbeat,row,column','sigma':args.sigma,'seed':args.seed,'checkpoint_path':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'timing_outliers':outliers,'training_coverage_note':'per-dimension bounds only; joint training coverage not proven','noise_mean':float(noise.mean()),'noise_std':float(noise.std()),'slices':meta,'assumptions':['real magnitude signals averaged at three box PSF midpoints','single pixel-center in-plane sampling','one representative event time per heartbeat','synthetic geometry','not k-space or slice-selective spin-history simulation']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));
    for name,count,*_ in definitions:
        d=np.load(out/(name+'.npz')); assert d['clean'].shape==(count,10,n,n) and d['noisy'].shape==d['clean'].shape and np.isfinite(d['noisy']).all()
    print('OUTPUT',out,flush=True)
if __name__=='__main__': main()
