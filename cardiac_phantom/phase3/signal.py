"""SCMR-v5 signal: explicitly bind the frozen protocol, never simulator defaults."""
import hashlib
import importlib.util
import sys
from pathlib import Path
import numpy as np
import torch

SOURCE = Path('/data2/dengyz/code/ismrm/ismrm_v2_3/src/ismrm_v2/proposed/trad_signal_decoder/trad_signal_simulator.py')
SOURCE_SHA256 = 'ea8b5a01c337e9601e95549057482a9d1170a2afef9c49e9a2cdb9c12f50707f'

def timing9(times_ms):
    times = np.asarray(times_ms, dtype=float)
    if times.shape != (10,) or not np.all(np.isfinite(times)) or not np.all(np.diff(times)>0):
        raise ValueError('HB1..HB10 must be strictly increasing finite times')
    result = np.diff(times) - 2.61*97
    result[3] -= 100
    result[6:] -= [35,45,55]
    if np.any(result < 0): raise ValueError('negative recovery interval')
    return result

def _simulator():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError('reference simulator hash differs from verified version')
    spec = importlib.util.spec_from_file_location('phase3_verified_trad', SOURCE)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module.TradSignalSimulator().eval(), module.TradProtocol(tr_ms=2.61, vps=87, ti_ms=(10.,100.))

def fingerprints(t1, t2, timing):
    t1,t2=np.broadcast_arrays(np.asarray(t1,dtype=np.float32),np.asarray(t2,dtype=np.float32))
    if np.any(~np.isfinite(t1)) or np.any(~np.isfinite(t2)) or np.any(t1<=0) or np.any(t2<=0): raise ValueError('only finite valid tissues allowed')
    simulator,protocol=_simulator(); flat1=t1.reshape(-1); flat2=t2.reshape(-1)
    tt=torch.from_numpy(np.broadcast_to(np.asarray(timing,dtype=np.float32),(len(flat1),9)).copy())
    with torch.no_grad():
        result=simulator(torch.from_numpy(flat1.copy()),torch.from_numpy(flat2.copy()),torch.ones(len(flat1)),tt,protocol,normalize=False)
    return result.numpy().reshape((*t1.shape,10))
