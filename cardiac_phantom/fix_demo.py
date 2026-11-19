import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from . import Phantom, sample_plane
from .export import grid, sample_grid, save_maps

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-root',default='runs'); args=ap.parse_args()
    out=Path(args.output_root)/datetime.now(timezone.utc).strftime('fix_%Y%m%dT%H%M%S_%fZ'); out.mkdir(parents=True)
    p=Phantom(); shape,aff=grid(); maps=sample_grid(p.query,shape,aff); save_maps(out/'static_3d',maps,aff)
    views=[('SAX z=12.37 mm',[0,0,12.37],[1,0,0],[0,1,0]),('4CH y=0 mm',[0,0,0],[1,0,0],[0,0,1]),('interface SAX',[30,0,12.37],[1,0,0],[0,1,0])]
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True); cm=ListedColormap(['#10121a','#df615d','#f5dc74','#68bde3'])
    for j,(name,center,col,row) in enumerate(views):
        step=.15 if j==2 else .75; n=200
        xyz,q=sample_plane(p,center,col,row,(step,step),(n,n)); f=p.fields(xyz); x,y,z=np.moveaxis(xyz,-1,0)
        old_rv=(((x-29)/29)**2+(y/25)**2+((z-1)/43)**2-1<=0)&((x/33)**2+(y/29)**2+(z/56)**2>1)
        old=np.where(f['lv_inner']<=0,2,np.where(f['lv_outer']<=0,1,np.where(old_rv,3,0)))
        for i,lab in enumerate((old,q['label'])):
            axes[i,j].imshow(lab,origin='lower',cmap=cm,vmin=0,vmax=3,interpolation='nearest',extent=[-step*n/2,step*n/2,-step*n/2,step*n/2]); axes[i,j].set(title=('baseline ' if i==0 else 'revised ')+name,xlabel='column offset (mm)',ylabel='row offset (mm)')
    fig.savefig(out/'geometry_comparison.png',dpi=150); plt.close(fig)
    xyz,q=sample_plane(p,[0,0,12.37],[1,0,0],[0,1,0],(.5,.5),(240,240))
    fig,axes=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
    for i,key in enumerate(('t1_ms','t2_ms')):
        for j,k in enumerate((1,2,3)):
            d=np.where(q['label']==k,100*(q[key]/p.tissues[k][i]-1),np.nan)
            im=axes[i,j].imshow(d,origin='lower',extent=[-60,60,-60,60],vmin=-5,vmax=5,cmap='coolwarm',interpolation='nearest'); axes[i,j].set(title=f'{key} tissue {k}',xlabel='x (mm)',ylabel='y (mm)'); fig.colorbar(im,ax=axes[i,j],label='% deviation')
    fig.savefig(out/'texture.png',dpi=150); plt.close(fig)
    stats={}
    for k in (1,2,3):
        stats[str(k)]={}
        for i,key in enumerate(('t1_ms','t2_ms')):
            v=maps[key][maps['label']==k]; stats[str(k)][key]={'min':float(v.min()),'max':float(v.max()),'mean':float(v.mean()),'std':float(v.std()),'relative_mean_deviation':float(v.mean()/p.tissues[k][i]-1)}
    info={'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'python':sys.executable,'versions':{'numpy':np.__version__,'matplotlib':matplotlib.__version__},'seed':p.seed,'amplitude':p.amplitude,'wavelength_mm':p.wavelength_mm,'shape_xyz':shape,'affine':aff.tolist(),'statistics':stats,'command':'python -m cardiac_phantom.fix_demo --output-root <new-root>'}
    (out/'summary.json').write_text(json.dumps(info,indent=2)+'\n'); print(out.resolve())
if __name__=='__main__': main()
