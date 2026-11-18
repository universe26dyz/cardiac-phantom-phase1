"""V10.10 golden-angle radial trajectory, 4 lines × 2 TR × 6 echoes."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class AcquisitionConfig:
    n: int = 128
    n_readout: int = 256
    n_motion_frames: int = 720
    tr_ms: float = 15.0
    te_ms_tr1: tuple = (2.38, 4.27, 6.16, 8.05, 9.94, 11.83)
    te_ms_tr2: tuple = (3.325, 5.215, 7.105, 8.995, 10.885, 12.775)

    @property
    def motion_frame_acquisition_ms(self):
        return 4 * 2 * self.tr_ms

    def validate(self):
        if self.n < 2 or self.n_readout < 2 or self.n_motion_frames < 1:
            raise ValueError("Invalid grid/readout/frame dimensions")
        te = np.asarray([self.te_ms_tr1, self.te_ms_tr2])
        if te.shape != (2, 6) or np.any(te <= 0) or np.any(te >= self.tr_ms):
            raise ValueError("TE must be 2 × 6 positive values below TR")
        if np.any(np.diff(te, axis=1) <= 0):
            raise ValueError("TE must be increasing within each TR")


def build_trajectory(cfg=None):
    cfg = cfg or AcquisitionConfig()
    cfg.validate()
    F = cfg.n_motion_frames
    line_ids = np.empty((F, 4), np.int32)
    line_ids[:, 0] = 0
    line_ids[:, 1:] = np.arange(1, F*3 + 1, dtype=np.int32).reshape(F, 3)
    navigators = line_ids == 0
    golden_angle = math.pi / ((1. + math.sqrt(5.)) / 2.)
    angles = np.mod(line_ids.astype(np.float64)*golden_angle, math.pi).astype(np.float32)
    angles[navigators] = 0.
    r = (np.arange(cfg.n_readout, dtype=np.float32) - cfg.n_readout//2) * np.float32(cfg.n / cfg.n_readout)
    kx = np.cos(angles[..., None]) * r[None, None, :]
    ky = np.sin(angles[..., None]) * r[None, None, :]
    ktraj = np.stack([kx, ky], axis=-1).astype(np.float32)
    te = np.asarray([cfg.te_ms_tr1, cfg.te_ms_tr2], np.float32)
    tr_start = np.empty((F, 4, 2), np.float32)
    echo_time = np.empty((F, 4, 2, 6), np.float32)
    for f in range(F):
        for l in range(4):
            for q in range(2):
                t0 = f*cfg.motion_frame_acquisition_ms + (l*2+q)*cfg.tr_ms
                tr_start[f, l, q] = t0
                echo_time[f, l, q] = t0 + te[q]
    return dict(line_ids=line_ids, is_navigator=navigators, angles_rad=angles,
                radius=r, ktraj_frames=ktraj, te_ms=te,
                tr_start_time_ms=tr_start, echo_time_ms=echo_time,
                event_index=np.arange(F*4*2*6, dtype=np.int64).reshape(F, 4, 2, 6))
