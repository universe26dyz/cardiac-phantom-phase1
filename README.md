# Continuous cardiac phantom — Phase 1

A synthetic, static ventricular model. It is intended for geometric and quantitative ground-truth experiments, not clinical anatomy or measured reference values. The existing V10.10 `lib/` and `test.ipynb` remain separate.

```python
from cardiac_phantom import Phantom, sample_plane
p = Phantom()
point = p.query([0., 0., 12.37])  # world coordinates, mm
xyz, maps = sample_plane(p, [0, 0, 12.37], [1, 0, 0], [0, 1, 0], (1, 1), (160, 160))
```

Requires Python, NumPy and Matplotlib; tests also use SciPy. From the repository root:

```bash
python -m unittest discover -s tests -v
MPLBACKEND=Agg python -m cardiac_phantom.demo --output-root runs
```

Each demo run creates a new timestamped directory, including anatomy and quantitative PNGs, native-value NPZ, and a JSON description. The NPZ uses `xyz_mm[row,column,xyz]`; all maps use `[row,column]`. Each pixel is evaluated at its center, with `(N-1)/2` at the plane center. Spacing is `(row, column)` in mm; column and row are unit vectors in world coordinates. Plots use lower origin: increasing coordinates point right and up, respectively. Summary JSON records these vectors and centers.

Local x, y, z form a right-handed coordinate system; positive z points apex to base. World coordinates follow `world = R @ local + translation`, with a proper orthonormal rotation; row-vector arrays use `local @ R.T + translation`. Default pose is identity. Both transformations accept `(...,3)`, including single points. Planes in the demo use identity pose. The default 2CH/4CH are ventricular views, not full multi-chamber anatomy.

The LV epicardium is a closed ellipsoid with semi-axes (32,28,55) mm; its contained blood pool is a closed ellipsoid with semi-axes (22,18,45) mm centered at z=2 mm. Thus both ends are closed, including the simplified base; the myocardium is the shell. The RV is an offset ellipsoid with the LV epicardial ellipsoid subtracted, leaving a crescent sharing the epicardial boundary. These dimensionless implicit fields are not distances. Queries evaluate them directly, without volume interpolation; labels and constants jump at interfaces.

| ID | Tissue | T1 (ms) | T2 (ms) | M0 (relative) |
|---|---|---:|---:|---:|
| 0 | background | NaN | NaN | 0 |
| 1 | LV myocardium | 1200 | 45 | 1 |
| 2 | LV blood | 1600 | 180 | 1 |
| 3 | RV blood | 1600 | 180 | 1 |

These are configurable synthetic target means in `DEFAULT_TISSUES`; `valid` excludes background from quantitative statistics. This version does not model atria, valves, RV myocardium, motion or realistic texture. Phase 2 motion is not implemented. Phase 3 will first compute each clean 10-weight observation including slice PSF/acquisition integration, then add zero-mean additive Gaussian noise at the observation level, with configurable strength and fixed seed, preserving clean and noisy inputs shared by both reconstruction methods. Noise is not applied to labels, ground truth or PSF subsamples. The reference protocol is TR 2.61 ms, VPS 87, 10 weights and SAX/2CH/4CH stacks of 12/15/10; these are not constraints on Phase 1 geometric sampling. No signal, noise or partial-volume simulation is implemented here.

## Phase 1 revision
The RV now uses the exact epicardial zero level as its clipping boundary. The boundary belongs to LV; RV occupies the strict exterior. No background layer is inserted. T1/T2 have independent per-tissue bounded Fourier textures, default seed 1234, amplitude ±5%, wavelength 20 mm (a Fourier wavelength, not measured correlation length). A fixed 30,000-point integration sample estimates centering; texture is defined in canonical local mm and independent of export grid. Set `Phantom(amplitude=0)` for constants. NIfTI exports are voxel-center samples in `(X,Y,Z)`, world mm; the original 160×160, 1 mm QC was 2D with no slice thickness or 3D resolution. Run `python -m cardiac_phantom.fix_demo --output-root <directory>` for a new run.

## Phase 2 motion
`Motion` exposes canonical-to-world, world-to-canonical, `query`, and world-mm forward `dvf`. Cardiac deformation continuously contracts local transverse radii by a Gaussian radial map (default 0.5 maximum, width 18 mm), and shortens z by up to 10%; a shared diffeomorphism moves all tissues, with milder displacement at greater radius. It is synthetic and not constrained to preserve myocardial volume. The monotone radial inverse uses 55 bisection iterations and raises if residual exceeds 1e-8 mm. The cardiac period defaults to 1 s; respiratory motion translates in synthetic world AP (world y, 4 mm) and SI (world z, 8 mm) with 4 s period. Both phases are configurable; displacements are anchored to zero at t=0. Cardiac deformation precedes respiratory translation, regardless of the static pose. Toggle either component through `cardiac` and `respiratory` flags.

Run `python -m cardiac_phantom.motion_demo --output-root <directory>` for fixed-world 2 mm `(X,Y,Z,T)` combined 4D maps, default 40 frames at 0.1 s, same-grid static maps, QC figures, tracked material-point motion and curves. T1/T2 are sampled at inverse-mapped canonical coordinates; they travel with tissue. JSON reports Jacobian-weighted reference-grid estimates, not respiratory raster counts, and lengths tracked at reference points. An optional dense forward world-basis DVF is written with `python -m cardiac_phantom.export_dvf <new-file.nii.gz> --time 0.5`; the vector dimension is NIfTI vector intent with axes `(X,Y,Z,1,3)`, not time.

## Phase 3 synthetic observations
Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m cardiac_phantom.phase3.full` to generate a new 37-slice run under `/data2/dengyz/dataset/phantom/phase3`. NPZ arrays are ordered slice, heartbeat, row, column; pose, times, PSF, checkpoint timing coverage and noise statistics are recorded in the manifest. These are real-valued magnitude-domain mixtures with three-point 8 mm box PSF and one event time per heartbeat, not complex k-space or slice-selective spin history. The synthetic MATLAB adapter `matlab/synthetic_scmr_v5.m` accepts explicit synthetic arrays and calls the frozen SCMR-v5 matcher with TI 10/100 ms; it does not claim clinical preprocessing. The traditional simulator source is hash-checked against its recorded server path.
