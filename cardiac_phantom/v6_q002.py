"""Run frozen Q002 while restoring its missing Q002-64 PSF chunk control.

The upstream 64-anchor controls omit a field later accessed by training. Set
the chunk size equal to the already specified eight PSF samples, so this is
one chunk and changes neither PSF samples nor the training objective.
"""
import os
from ismrm_v2.proposed.q002 import training
from ismrm_v2.proposed.q002.run_q002_reconstruction import main

_original = training.q002_training_controls

def _controls(*args, **kwargs):
    values = _original(*args, **kwargs)
    if values['anchor_batch_size'] == 64:
        values['psf_forward_chunk_size'] = values['psf_samples']
    return values

training.q002_training_controls = _controls
if __name__ == '__main__':
    os.chdir('/data2/dengyz/code/ismrm/ismrm_v2_3')
    main()
