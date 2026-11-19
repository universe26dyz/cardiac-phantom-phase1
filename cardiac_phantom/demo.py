"""Generate a new, non-overwriting Phase 1 QC run directory."""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from . import Phantom, NAMES, sample_plane


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', default='runs')
    args = parser.parse_args()
    run = Path(args.output_root) / datetime.now(timezone.utc).strftime('phase1_%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True, exist_ok=False)
    p = Phantom()
    s = 1 / np.sqrt(2)
    views = {
        'SAX apex z=-35 mm': ([0, 0, -35], [1, 0, 0], [0, 1, 0]),
        'SAX mid z=12.37 mm': ([0, 0, 12.37], [1, 0, 0], [0, 1, 0]),
        'SAX base z=35 mm': ([0, 0, 35], [1, 0, 0], [0, 1, 0]),
        '2CH x=0 mm': ([0, 0, 0], [0, 1, 0], [0, 0, 1]),
        '4CH y=0 mm': ([0, 0, 0], [1, 0, 0], [0, 0, 1]),
        'Oblique': ([0, 0, 5], [s, s, 0], [-0.5, 0.5, s]),
    }
    shape, spacing = (160, 160), (1., 1.)
    data = {name: sample_plane(p, *pose, spacing, shape) for name, pose in views.items()}
    colors = ['#10121a', '#df615d', '#f5dc74', '#68bde3']
    cmap = ListedColormap(colors)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for ax, (name, (_, result)) in zip(axes.flat, data.items()):
        ax.imshow(result['label'], origin='lower', extent=[-80,80,-80,80], cmap=cmap, norm=BoundaryNorm(np.arange(5)-.5,4), interpolation='nearest')
        ax.set(title=name, xlabel='column direction (mm)', ylabel='row direction (mm)')
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=colors[i], label=NAMES[i]) for i in range(4)], loc='outside lower center', ncol=4)
    fig.savefig(run/'anatomy.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for row, name in enumerate(('SAX mid z=12.37 mm', '4CH y=0 mm')):
        for col, (key, unit, limits) in enumerate((('t1_ms', 'ms', (0,1800)), ('t2_ms', 'ms', (0,200)), ('m0', 'relative', (0,1.1)))):
            im = axes[row,col].imshow(data[name][1][key], origin='lower', extent=[-80,80,-80,80], vmin=limits[0], vmax=limits[1], interpolation='nearest')
            axes[row,col].set(title=f'{name}: {key}', xlabel='column direction (mm)', ylabel='row direction (mm)')
            fig.colorbar(im, ax=axes[row,col], label=unit)
    fig.savefig(run/'quantitative.png', dpi=150); plt.close(fig)
    xyz, result = data['SAX mid z=12.37 mm']
    np.savez_compressed(run/'sax_mid.npz', xyz_mm=xyz, **result)
    summary = {'coordinate_system': 'world mm; identity pose', 'spacing_row_column_mm': spacing, 'shape_row_column': shape,
               'tissues': {str(k): {'name': NAMES[k], 't1_ms': None if k == 0 else v[0], 't2_ms': None if k == 0 else v[1], 'm0_relative': v[2]} for k,v in p.tissues.items()},
               'views': {name: {'center_mm': pose[0], 'column_direction': pose[1], 'row_direction': pose[2], 'label_counts': {str(k): int(np.count_nonzero(data[name][1]['label'] == k)) for k in NAMES}} for name,pose in views.items()},
               'python_executable': sys.executable, 'python_version': sys.version, 'numpy_version': np.__version__, 'matplotlib_version': matplotlib.__version__,
               'git_head': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(), 'command': 'python -m cardiac_phantom.demo --output-root runs'}
    (run/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(run.resolve())

if __name__ == '__main__':
    main()
