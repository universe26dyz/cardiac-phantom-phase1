"""Fresh Q002 experiment with both stage iteration counts doubled.

All model, optimizer, data and PSF settings follow v6_q002. The original
control validator runs first; only the returned stage lengths and the missing
64-anchor PSF chunk key are changed. Scheduler milestones remain at the same
fraction of each stage, so their absolute iteration numbers also double.
"""
import os
from ismrm_v2.proposed.q002 import training
from ismrm_v2.proposed.q002.run_q002_reconstruction import main

_original = training.q002_training_controls

def _long_controls(*args, **kwargs):
    values = _original(*args, **kwargs)
    if values['anchor_batch_size'] != 64:
        raise ValueError('This experiment requires the original 64-anchor Q002 route')
    values['stage_a_iterations'] = 4000
    values['stage_b_iterations'] = 8000
    values['psf_forward_chunk_size'] = values['psf_samples']
    return values

training.q002_training_controls = _long_controls
if __name__ == '__main__':
    os.chdir('/data2/dengyz/code/ismrm/ismrm_v2_3')
    main()
