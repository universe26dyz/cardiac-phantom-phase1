"""V10.10 synthetic acquisition: event-time material-point Type-3 Fourier signal.

Output kspace: [selected_frame, line(4), TR(2), echo(6), readout] complex64.
The ``echo_time`` mode linearly interpolates only the *ground-truth motion*
anchors, just as the original V10.10 implementation does. No reconstruction.
"""
import importlib.util
import numpy as np
from .signal import angular_radial_b0_hz, spectral_signal


def _fourier(coords, source, kline, backend='direct', chunk=128, eps=1e-5):
    if backend == 'finufft':
        import finufft
        x = np.ascontiguousarray(coords[:, 0], dtype=np.float32)
        y = np.ascontiguousarray(coords[:, 1], dtype=np.float32)
        kx = np.ascontiguousarray(2*np.pi*kline[:, 0], dtype=np.float32)
        ky = np.ascontiguousarray(2*np.pi*kline[:, 1], dtype=np.float32)
        v = np.asarray(source, np.complex64)
        if v.ndim == 1:
            return np.asarray(finufft.nufft2d3(x, y, np.ascontiguousarray(v), kx, ky,
                                               isign=-1, eps=eps), np.complex64)
        return np.stack([_fourier(coords, v[:, e], kline, 'finufft', eps=eps)
                         for e in range(v.shape[1])], axis=-1)
    if backend != 'direct':
        raise ValueError('backend must be direct|finufft|auto')
    values = np.asarray(source, np.complex64)
    out = []
    for s in range(0, len(kline), int(chunk)):
        ph = -2*np.pi*(kline[s:s+int(chunk)] @ coords.T)
        out.append(np.exp(1j*ph).astype(np.complex64) @ values)
    return np.concatenate(out, axis=0).astype(np.complex64)


def _coords_at_ms(gt, ms, frame_ms):
    F = gt.shape[2]
    pos = max(0., float(ms))/float(frame_ms)
    lo = min(int(np.floor(pos)), F-1)
    if lo >= F - 1:
        return gt[:, :, F-1, :2]
    alpha = np.float32(pos - lo)
    return ((np.float32(1.)-alpha)*gt[:, :, lo, :2] +
            alpha*gt[:, :, lo+1, :2])


def simulate_acquisition(gt, maps, trajectory, *,
                         frame_indices=None, motion_time_sampling='echo_time',
                         motion_frame_duration_ms=120., b0_peak_hz=150.,
                         larmor_hz_per_ppm=128., backend='auto',
                         direct_k_chunk=128, finufft_eps=1e-5):
    """Synthesize V10.10 acquired complex k-space (no training/reconstruction).

    backend='auto': prefer optional FINUFFT, otherwise direct NumPy NUDFT.
    Source points are the active material pixels (rho>0); no image gridding.
    """
    if motion_time_sampling not in ('echo_time', 'tr_start', 'frame_held'):
        raise ValueError('motion_time_sampling must be echo_time|tr_start|frame_held')
    if backend == 'auto':
        backend = 'finufft' if importlib.util.find_spec('finufft') else 'direct'
    if backend == 'finufft' and not importlib.util.find_spec('finufft'):
        raise ImportError('Install finufft, or use backend="direct"')
    if motion_frame_duration_ms <= 0:
        raise ValueError('motion_frame_duration_ms must be positive')

    gt = np.asarray(gt, dtype=np.float32)
    nframes = gt.shape[2]
    idx = (np.arange(nframes, dtype=np.int32) if frame_indices is None
           else np.asarray(frame_indices, dtype=np.int32))
    if np.any(idx < 0) or np.any(idx >= nframes):
        raise IndexError('frame_indices outside ground-truth motion range')
    ktraj = np.asarray(trajectory['ktraj_frames'], np.float32)
    te = np.asarray(trajectory['te_ms'], np.float32)
    t_start = np.asarray(trajectory['tr_start_time_ms'], np.float32)
    t_echo = np.asarray(trajectory['echo_time_ms'], np.float32)
    if ktraj.shape[0] != nframes:
        raise ValueError('Trajectory and ground truth must share n_motion_frames')
    out = np.empty((len(idx), 4, 2, 6, ktraj.shape[2]), np.complex64)
    active = maps['rho'] > 0
    material = [np.asarray(maps[key], np.float32)[active]
                for key in ('rho', 'ff', 'nmidb', 'ddb', 'f0_rad', 't2star_ms')]

    for oi, f in enumerate(idx):
        if motion_time_sampling == 'frame_held':
            coords_frame = gt[:, :, f, :2][active].reshape(-1, 2)
            b0_frame = angular_radial_b0_hz(coords_frame, b0_peak_hz)
        for l in range(4):
            kline = ktraj[f, l]
            for q in range(2):
                if motion_time_sampling in ('frame_held', 'tr_start'):
                    if motion_time_sampling == 'frame_held':
                        coords, b0 = coords_frame, b0_frame
                    else:
                        coords = _coords_at_ms(gt, t_start[f, l, q], motion_frame_duration_ms)[active].reshape(-1, 2)
                        b0 = angular_radial_b0_hz(coords, b0_peak_hz)
                    sig = np.stack([spectral_signal(*material, b0, float(te[q, e]),
                                                    larmor_hz_per_ppm)
                                    for e in range(6)], axis=-1)
                    out[oi, l, q] = _fourier(coords, sig, kline, backend, direct_k_chunk, finufft_eps).T
                else:
                    for e in range(6):
                        coords = _coords_at_ms(gt, t_echo[f, l, q, e], motion_frame_duration_ms)[active].reshape(-1, 2)
                        b0 = angular_radial_b0_hz(coords, b0_peak_hz)
                        sig = spectral_signal(*material, b0, float(te[q, e]), larmor_hz_per_ppm)
                        out[oi, l, q, e] = _fourier(coords, sig, kline, backend, direct_k_chunk, finufft_eps)
    return out
