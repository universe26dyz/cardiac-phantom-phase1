"""Regenerate the 512x512, 1.5 mm, 2x heart, mirrored L100 phantom."""
import argparse, json, shutil, subprocess
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cardiac_phantom import Phantom
from cardiac_phantom.motion import Motion
from cardiac_phantom.phase3.signal import fingerprints, timing9
from cardiac_phantom.phase3.psf import integrate

class ScaledMirroredPhantom:
    """Scale all local phantom coordinates and reflect the local x axis."""
    def __init__(self, base, scale=2.0):
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.base, self.scale = base, float(scale)
    def _inverse_transform(self, p):
        q=np.asarray(p,dtype=float).copy()
        q[...,0] /= -self.scale
        q[...,1:] /= self.scale
        return q
    def local_to_world(self, xyz):
        q=np.asarray(xyz,dtype=float).copy()
        q[...,0] *= -self.scale
        q[...,1:] *= self.scale
        return self.base.local_to_world(q)
    def world_to_local(self, xyz):
        return self._inverse_transform(self.base.world_to_local(xyz))
    def query(self, xyz):
        local=self._inverse_transform(self.base.world_to_local(xyz))
        return self.base.query(self.base.local_to_world(local))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-root",default="/data2/dengyz/dataset/phantom/v6")
    ap.add_argument("--size",type=int,default=512)
    ap.add_argument("--spacing-mm",type=float,default=1.5)
    ap.add_argument("--heart-scale",type=float,default=2.0)
    ap.add_argument("--seed",type=int,default=123)
    ap.add_argument("--sigma",type=float,default=.002)
    a=ap.parse_args()
    if (a.size,a.spacing_mm,a.heart_scale)!=(512,1.5,2.0):
        raise ValueError("This regeneration is frozen at 512x512, 1.5 mm, 2x heart")
    base=Phantom()
    phantom=ScaledMirroredPhantom(base,a.heart_scale)
    # Verify RV is on negative x, LV cavity remains centered.
    labels=phantom.query(np.array([[-80.,0,0],[80.,0,0],[0.,0,0]]))["label"].tolist()
    if labels != [3,0,2]:
        raise RuntimeError(f"mirrored chamber check failed (RV-, empty+, LV cavity): {labels}")
    motion=Motion(phantom=phantom); rng=np.random.default_rng(a.seed)
    stamp=datetime.now(timezone.utc).strftime("l100_pilot_heart2x_mirrorx_%Y%m%dT%H%M%S_%fZ")
    out=Path(a.output_root)/stamp
    out.mkdir(parents=True,exist_ok=False)
    for case in ("static","respiratory"):
        (out/case).mkdir()
    n=a.size; scale=a.heart_scale; spacing=a.spacing_mm; rr=1000.0
    offsets=np.array([-8/3,0,8/3]); psf_weights=np.full(3,1/3)
    grid=(np.arange(n)-(n-1)/2)*spacing
    r,c=np.meshgrid(grid,grid,indexing="ij")
    meta=[]
    definitions=[
      ("sax",12,[0,0,1],[0,1,0],[1,0,0],np.linspace(-40,40,12)),
      ("2ch",15,[1,0,0],[0,0,1],[0,1,0],np.linspace(-20,20,15)),
      ("4ch",10,[0,1,0],[0,0,1],[1,0,0],np.linspace(-18,18,10))]
    for si,(name,count,normal,row,col,positions) in enumerate(definitions):
        normal,row,col=map(lambda v:np.asarray(v,float),(normal,row,col))
        clean={k:[] for k in ("static","respiratory")}
        noisy={k:[] for k in clean}; noise_list=[]
        for j,pos in enumerate(positions*scale):
            center=normal*pos
            xyz=center[None,None,None,:]+r[None,:,:,None]*row[None,None,None,:]+c[None,:,:,None]*col[None,None,None,:]+offsets[:,None,None,None]*normal[None,None,None,:]
            times=(si*20+j)*10300+np.arange(10)*rr
            timing=timing9(times)
            obs={k:np.zeros((10,n,n),np.float32) for k in clean}
            for hb,t in enumerate(times):
                for case in clean:
                    q=motion.query(xyz,t/1000,cardiac=False,respiratory=(case=="respiratory"))
                    valid=q["valid"]
                    signal=np.zeros(valid.shape,np.float32)
                    if valid.any():
                        signal[valid]=fingerprints(q["t1_ms"][valid],q["t2_ms"][valid],timing)[:,hb]*q["m0"][valid]
                    obs[case][hb]=integrate(signal,psf_weights).astype(np.float32)
            noise=rng.normal(0,a.sigma,(10,n,n)).astype("f4")
            noise_list.append(noise)
            for case in clean:
                clean[case].append(obs[case])
                noisy[case].append((obs[case]+noise).astype("f4"))
            meta.append({"stack":name,"slice_id":j,"center_mm":center.tolist(),
              "row_unit":row.tolist(),"column_unit":col.tolist(),"normal_unit":normal.tolist(),
              "shape_rc":[n,n],"spacing_rc_mm":[spacing,spacing],"slice_thickness_mm":8,
              "psf_offsets_mm":offsets.tolist(),"psf_weights":psf_weights.tolist(),
              "acquisition_time_ms":times.tolist(),"timing9_ms":timing.tolist(),
              "cardiac_phase_cycles":[0.0]*10,
              "respiratory_phase_cycles":((times/1000)%motion.respiratory_period/motion.respiratory_period).tolist()})
            print(name,j+1,"/",count,flush=True)
        for case in clean:
            clean[case]=np.asarray(clean[case],dtype=np.float32)
            noisy[case]=np.asarray(noisy[case],dtype=np.float32)
            np.savez_compressed(out/case/(name+".npz"),clean=clean[case],noisy=noisy[case],noise=np.asarray(noise_list))
        # Phase 3 style 10-weight clean midpoint mosaics with a shared scale.
        vmax=float(max(clean["static"].max(),clean["respiratory"].max()))
        for case in clean:
            mid=count//2
            fig,axes=plt.subplots(2,5,figsize=(12,5))
            for hb,ax in enumerate(axes.flat):
                ax.imshow(clean[case][mid,hb],origin="lower",cmap="gray",vmin=0,vmax=vmax)
                ax.set_title(f"HB{hb+1}"); ax.axis("off")
            fig.suptitle(f"{case} {name.upper()} clean midpoint | mirrored RV, heart x{scale:g}")
            fig.tight_layout(); fig.savefig(out/f"{case}_{name}_montage.png",dpi=150); plt.close(fig)
        mid=count//2; hb=4
        vmax_pair=float(max(clean["static"][mid,hb].max(),clean["respiratory"][mid,hb].max()))
        residual=noisy["respiratory"][mid,hb]-clean["respiratory"][mid,hb]
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        axes[0].imshow(clean["static"][mid,hb],origin="lower",cmap="gray",vmin=0,vmax=vmax_pair); axes[0].set_title("static")
        axes[1].imshow(clean["respiratory"][mid,hb],origin="lower",cmap="gray",vmin=0,vmax=vmax_pair); axes[1].set_title("respiratory motion")
        im=axes[2].imshow(residual,origin="lower",cmap="coolwarm",vmin=-.006,vmax=.006); axes[2].set_title("noise residual")
        for ax in axes: ax.set_xlabel("column"); ax.set_ylabel("row")
        fig.suptitle(f"{name.upper()} slice {mid}, HB5 | {n}x{n}, {spacing:g} mm")
        fig.tight_layout(); fig.savefig(out/f"{name}_comparison.png",dpi=150); plt.close(fig)
    # Verify the generated PSF sample exactly matches the stored SAX clean slice.
    v=meta[6]; center=np.asarray(v["center_mm"]); row=np.asarray(v["row_unit"]); col=np.asarray(v["column_unit"]); normal=np.asarray(v["normal_unit"])
    t=v["acquisition_time_ms"][4]; timing=np.asarray(v["timing9_ms"])
    xyz=center+r[...,None]*row+c[...,None]*col
    q=motion.query(xyz,t/1000,cardiac=False,respiratory=False); point=np.zeros((n,n),np.float32); valid=q["valid"]
    point[valid]=fingerprints(q["t1_ms"][valid],q["t2_ms"][valid],timing)[:,4]*q["m0"][valid]
    xyzp=center[None,None,None,:]+r[None,:,:,None]*row[None,None,None,:]+c[None,:,:,None]*col[None,None,None,:]+offsets[:,None,None,None]*normal[None,None,None,:]
    q=motion.query(xyzp,t/1000,cardiac=False,respiratory=False); sig=np.zeros(q["valid"].shape,np.float32); valid=q["valid"]
    sig[valid]=fingerprints(q["t1_ms"][valid],q["t2_ms"][valid],timing)[:,4]*q["m0"][valid]
    thick=integrate(sig,psf_weights).astype(np.float32)
    saved=np.load(out/"static/sax.npz")["clean"][6,4]
    psf_error=float(np.max(np.abs(thick-saved)))
    if psf_error>1e-6: raise RuntimeError(f"PSF verification failed: {psf_error}")
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,img,title in zip(axes,(point,thick,thick-point),("zero thickness","8 mm box PSF","PSF difference")):
        obj=ax.imshow(img,origin="lower",cmap="gray" if title!="PSF difference" else "coolwarm")
        ax.set_title(title); ax.set_xlabel("column"); ax.set_ylabel("row")
    fig.colorbar(obj,ax=axes[-1],fraction=.046,pad=.04)
    fig.suptitle(f"SAX slice ID 6, HB5 | sampling PSF | {n}x{n}, {spacing:g} mm")
    fig.tight_layout(); fig.savefig(out/"psf_comparison.png",dpi=150); plt.close(fig)
    shutil.copy2(__file__,out/"generate_phantom.py")
    manifest={"label":"L100 512x512 high-resolution mirrored, 2x-heart synthetic phantom; no reconstruction/training",
      "generator_git_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
      "array_order":"slice,heartbeat,row,column",
      "geometry":{"matrix_rc":[n,n],"spacing_rc_mm":[spacing,spacing],"fov_rc_mm":[n*spacing,n*spacing],
        "heart_geometry_scale":scale,"ventricular_reflection":"local x; RV from +x to -x in SAX display",
        "slice_center_positions_scaled":True,"slice_thickness_mm":8},
      "motion":{"cardiac":"disabled; ED phase 0","respiratory":"sinusoidal global AP/SI translation only",
        "respiratory_period_s":motion.respiratory_period,"ap_mm":motion.respiratory_ap_mm,"si_mm":motion.respiratory_si_mm},
      "protocol":{"TI_ms":[10,100],"TR_ms":2.61,"VPS":87,"RR_ms":rr},
      "sampling_psf":{"type":"three-point box integral through slice thickness","offsets_mm":offsets.tolist(),"weights":psf_weights.tolist()},
      "noise":{"type":"paired additive Gaussian after PSF","seed":a.seed,"sigma":a.sigma},
      "mind_registration":"not applied; raw generated observations",
      "sanity_check_labels_at_xyz_mm":{"RV_minus_x":int(labels[0]),"outside_plus_x":int(labels[1]),"LV_cavity_center":int(labels[2])},
      "psf_max_abs_error_vs_saved_clean":psf_error,"slices":meta}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("OUTPUT",out,"PSF_MAX_ABS_ERROR",psf_error,"LABELS",labels,flush=True)
if __name__=="__main__": main()
