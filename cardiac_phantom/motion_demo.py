import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from .motion import Motion
from . import sample_plane
from .export import grid, sample_grid, save_maps, KEYS

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-root',default='runs'); ap.add_argument('--dt',type=float,default=.1); ap.add_argument('--frames',type=int,default=40); ap.add_argument('--spacing',type=float,default=2.); args=ap.parse_args()
    out=Path(args.output_root)/datetime.now(timezone.utc).strftime('motion_%Y%m%dT%H%M%S_%fZ'); out.mkdir(parents=True)
    m=Motion(); shape,aff=grid(args.spacing); static=sample_grid(m.phantom.query,shape,aff); save_maps(out/'static_3d',static,aff)
    tmp=out/'scratch'; tmp.mkdir(); arrays={k:np.memmap(tmp/(k+'.dat'),mode='w+',dtype=np.uint8 if k in ('label','valid') else np.float32,shape=(*shape,args.frames)) for k in KEYS}
    times=np.arange(args.frames)*args.dt
    for i,t in enumerate(times):
        values=static if i==0 else sample_grid(lambda x:m.query(x,t),shape,aff)
        if any(np.any(np.take(values['label'], j, axis=ax)) for ax in range(3) for j in (0,-1)):
            raise RuntimeError(f'FOV truncates tissue at frame {i}')
        for k in KEYS: arrays[k][...,i]=values[k]
        print(f'frame {i+1}/{args.frames}',flush=True)
    save_maps(out/'combined_4d',arrays,aff,args.dt)
    del values,arrays
    for k in KEYS:
        img=nib.load(out/'combined_4d'/(k+'.nii.gz')); ref=nib.load(out/'static_3d'/(k+'.nii.gz'))
        if img.shape != (*shape,args.frames) or img.header.get_xyzt_units() != ('mm','sec') or not np.allclose(img.affine,aff) or not np.isclose(img.header.get_zooms()[3],args.dt): raise RuntimeError('4D header mismatch')
        np.testing.assert_allclose(np.asarray(img.dataobj)[...,0],np.asarray(ref.dataobj),equal_nan=True)
    for p in tmp.iterdir(): p.unlink()
    tmp.rmdir()
    cmap=ListedColormap(['#10121a','#df615d','#f5dc74','#68bde3']); views=[('SAX',[0,0,12.37],[1,0,0],[0,1,0]),('2CH',[0,0,0],[0,1,0],[0,0,1]),('4CH',[0,0,0],[1,0,0],[0,0,1])]
    for name,flags in [('cardiac',(True,False)),('respiratory',(False,True)),('combined',(True,True))]:
        fig,axes=plt.subplots(3,3,figsize=(12,10),constrained_layout=True)
        for i,(vn,center,col,row) in enumerate(views):
            xyz,_=sample_plane(m.phantom,center,col,row,(1,1),(140,140))
            for j,t in enumerate((0,.5,1.)):
                q=m.query(xyz,t,*flags); axes[i,j].imshow(q['label'],origin='lower',cmap=cmap,vmin=0,vmax=3,interpolation='nearest',extent=[-70,70,-70,70]); axes[i,j].set(title=f'{vn} {name} t={t}s',xlabel='column (mm)',ylabel='row (mm)')
        fig.savefig(out/(name+'.png'),dpi=130); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    xyz,_=sample_plane(m.phantom,[0,0,12.37],[1,0,0],[0,1,0],(.75,.75),(180,180))
    for ax,t in zip(axes,(0,.5,1.)):
        q=m.query(xyz,t); dev=np.full(q['label'].shape,np.nan)
        for k in (1,2,3):
            mask=q['label']==k; dev[mask]=100*(q['t1_ms'][mask]/m.phantom.tissues[k][0]-1)
        im=ax.imshow(dev,origin='lower',extent=[-67.5,67.5,-67.5,67.5],vmin=-5,vmax=5,cmap='coolwarm'); ax.contour(q['label'],levels=[.5,1.5,2.5],colors='k',linewidths=.4,origin='lower',extent=[-67.5,67.5,-67.5,67.5]); ax.set(title=f'combined t={t}s',xlabel='x (mm)',ylabel='y (mm)')
    fig.colorbar(im,ax=axes,label='T1 deviation (%)'); fig.savefig(out/'texture_motion.png',dpi=150); plt.close(fig)
    # Material-volume integration using fixed 2 mm reference grid and Jacobian determinant.
    idx=np.argwhere(static['label']>0); pts=nib.affines.apply_affine(aff,idx); labels=static['label'][tuple(idx.T)]; eps=1e-3; metrics=[]
    for t in times:
        jac=np.stack([(m.canonical_to_world(pts+np.eye(3)[i]*eps,t,True,False)-m.canonical_to_world(pts-np.eye(3)[i]*eps,t,True,False))/(2*eps) for i in range(3)],axis=-1); det=np.linalg.det(jac)
        inn=np.array([22.,0.,2.]); outer=np.array([32*np.sqrt(1-(2/55)**2),0.,2.]); tr=m.canonical_to_world(np.array([inn,outer]),t,True,False)
        ap=m.canonical_to_world(np.array([[0,0,-55],[0,0,55]]),t,True,False)
        metrics.append({'t_s':float(t),'cardiac_phase_cycles':float(t/m.cardiac_period%1),'respiratory_phase_cycles':float(t/m.respiratory_period%1),'volumes_mm3':{str(k):float(det[labels==k].sum()*args.spacing**3) for k in (1,2,3)},'long_axis_mm':float(np.linalg.norm(ap[1]-ap[0])),'wall_thickness_mm':float(np.linalg.norm(tr[1]-tr[0])),'min_sampled_jacobian':float(det.min())})
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for k,n in ((1,'LV myocardium'),(2,'LV blood'),(3,'RV blood')): axes[0].plot(times,[v['volumes_mm3'][str(k)]/1000 for v in metrics],label=n)
    axes[0].set(xlabel='time (s)',ylabel='volume (mL)',title='Cardiac deformation'); axes[0].legend()
    axes[1].plot(times,[v['long_axis_mm'] for v in metrics]); axes[1].set(xlabel='time (s)',ylabel='length (mm)',title='Tracked epicardial long axis')
    axes[2].plot(times,[v['wall_thickness_mm'] for v in metrics]); axes[2].set(xlabel='time (s)',ylabel='thickness (mm)',title='Tracked radial wall')
    fig.savefig(out/'motion_curves.png',dpi=150); plt.close(fig)
    samples=np.array([[0,0,0],[26,0,0],[45,0,0],[0,0,-45],[20,5,10]],float); tracks=[]
    for t in (0,.25,.5,1.,2.,4.):
        world=m.canonical_to_world(samples,t); tracks.append({'t_s':t,'cardiac_phase_cycles':t/m.cardiac_period%1,'respiratory_phase_cycles':t/m.respiratory_period%1,'canonical_mm':samples.tolist(),'world_mm':world.tolist(),'forward_dvf_world_mm':m.dvf(samples,t).tolist()})
    (out/'motion_gt.json').write_text(json.dumps(tracks,indent=2));
    sizes={str(p.relative_to(out)):p.stat().st_size for p in out.rglob('*.nii.gz')}
    info={'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'python':sys.executable,'versions':{'numpy':np.__version__,'nibabel':nib.__version__,'matplotlib':matplotlib.__version__},'grid_shape_xyzt':[*shape,args.frames],'affine':aff.tolist(),'dt_s':args.dt,'t1_t2_unit':'ms','world_axes':'x right, y AP, z SI (synthetic)','metrics':metrics,'metric_method':'reference 2 mm midpoint samples weighted by cardiac Jacobian; respiration preserves volume; length and wall from tracked material points','files_bytes':sizes,'command':'python -m cardiac_phantom.motion_demo --output-root <directory>'}
    (out/'summary.json').write_text(json.dumps(info,indent=2)); print(out.resolve())
if __name__=='__main__': main()
