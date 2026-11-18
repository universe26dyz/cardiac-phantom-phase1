"""Original V5/V10.10 material-point circular phantom and signal maps (NumPy only).

Ground truth layout: gt[y, x, frame, :] = [x(t), y(t), M0].
Coordinates are FOV-normalized: one pixel corresponds to 1/n.
Motion anchors follow the original variable-length cycle generator.
"""
from dataclasses import dataclass
import numpy as np
from .signal import angular_radial_b0_hz

DEFAULT_CYCLE_LENGTHS = (40, 42, 38, 40, 44, 36, 40, 46, 34,
                         40, 46, 34, 40, 44, 36, 40, 44, 36)


@dataclass(frozen=True)
class PhantomConfig:
    n: int = 128
    cycle_lengths: tuple = DEFAULT_CYCLE_LENGTHS
    outer_radius_px: tuple = (30., 40., 50., 60.)
    max_contraction_pixels: float = 5.0
    nominal_cycle_seconds: float = 2.0

    @property
    def n_frames(self):
        return int(sum(self.cycle_lengths))


@dataclass(frozen=True)
class SignalConfig:
    rho_values: tuple = (100., 75., 125., 55.)
    ff_values: tuple = (0.05, 0.25, 0.65, 0.85)
    nmidb_values: tuple = (0.05, 0.20, 0.35, 0.55)
    ddb_values: tuple = (0.15, 1.20, 1.55, 1.95)
    t2star_ms_values: tuple = (30., 30., 30., 30.)
    b0_peak_hz: float = 150.0
    larmor_hz_per_ppm: float = 128.0
    f0_phase_cycles_x: float = 0.12
    f0_phase_cycles_y: float = -0.08


def _radial_phases(phase, r, theta, n, max_pixels):
    """Same dual-cycle radial displacement template as original lib/phantom.py.

    Deliberately retain the original fixed 30/40/50/60-pixel motion regions;
    outer_radius_px defines material rings, not the motion-region boundaries.
    """
    t = (2.0 * float(phase)) % 2.0
    outer = -max_pixels * t / .5 if t < .5 else (
        -max_pixels if t < 1. else (
        -max_pixels * (1. - (t - 1.) / .5) if t < 1.5 else 0.))
    ti = t % 1.0
    inner = (0.0 if ti < .5 else
             -max_pixels * (ti - .5) / .1 if ti < .6 else
             -max_pixels if ti < .9 else
             -max_pixels * (1. - (ti - .9) / .1))
    dr = np.zeros_like(r, dtype=np.float64)
    mask = r <= 30.
    dr[mask] = inner * r[mask] / 30.
    mask = (r > 30.) & (r <= 40.)
    dr[mask] = inner
    mask = (r > 40.) & (r <= 50.)
    w = (r[mask] - 40.) / 10.
    dr[mask] = (1. - w) * inner + w * outer
    mask = (r > 50.) & (r <= 60.)
    dr[mask] = outer
    mask = r > 60.
    if np.any(mask):
        center = (n - 1.) / 2.
        boundary = np.minimum(center / np.maximum(np.abs(np.cos(theta)), 1e-6),
                              center / np.maximum(np.abs(np.sin(theta)), 1e-6))
        w = np.clip((r[mask] - 60.) / np.maximum(boundary[mask] - 60., 1e-6), 0., 1.)
        dr[mask] = (1. - w) * outer
    return dr


def _ring_map(radius, active, radii, vals):
    out = np.zeros(radius.shape, np.float32)
    cuts = (0.,) + tuple(float(x) for x in radii)
    for j in range(4):
        mask = (radius > cuts[j]) & (radius <= cuts[j + 1]) & active
        if j == 0:
            mask = (radius <= cuts[1]) & active
        out[mask] = vals[j]
    return out


def build_phantom(config=None, signal_config=None):
    """Return (gt, maps, metadata). Motion GT shape is [n,n,frames,3]."""
    cfg = config or PhantomConfig()
    sc = signal_config or SignalConfig()
    n = cfg.n
    if n < 2 or len(cfg.outer_radius_px) != 4 or any(np.diff(cfg.outer_radius_px) <= 0):
        raise ValueError("n >= 2 and four increasing material radii required")
    if not cfg.cycle_lengths or any(int(x) <= 0 for x in cfg.cycle_lengths):
        raise ValueError("cycle_lengths must contain positive integers")
    axis = np.arange(n, dtype=np.float64) - (n - 1.) / 2.
    yy_px, xx_px = np.meshgrid(axis, axis, indexing='ij')
    radius = np.sqrt(xx_px**2 + yy_px**2)
    theta = np.arctan2(yy_px, xx_px)
    active = radius <= float(cfg.outer_radius_px[-1])
    fov = np.stack((xx_px / n, yy_px / n), axis=-1).astype(np.float32)

    phases = np.concatenate([np.arange(length, dtype=np.float64) / float(length)
                             for length in cfg.cycle_lengths]).astype(np.float32)
    gt = np.empty((n, n, len(phases), 3), dtype=np.float32)
    rho = _ring_map(radius, active, cfg.outer_radius_px, sc.rho_values)
    for f, phase in enumerate(phases):
        dr = _radial_phases(float(phase), radius, theta, n, cfg.max_contraction_pixels)
        rnew = np.maximum(radius + dr, 0.)
        scale = np.ones_like(radius)
        mask = radius > 1e-7
        scale[mask] = rnew[mask] / radius[mask]
        gt[:, :, f, 0] = (xx_px * scale / n).astype(np.float32)
        gt[:, :, f, 1] = (yy_px * scale / n).astype(np.float32)
        gt[:, :, f, 2] = rho

    ff = _ring_map(radius, active, cfg.outer_radius_px, sc.ff_values)
    nmidb = _ring_map(radius, active, cfg.outer_radius_px, sc.nmidb_values)
    ddb = _ring_map(radius, active, cfg.outer_radius_px, sc.ddb_values)
    t2star = _ring_map(radius, active, cfg.outer_radius_px, sc.t2star_ms_values)
    f0 = (2 * np.pi * (sc.f0_phase_cycles_x * fov[..., 0] +
                        sc.f0_phase_cycles_y * fov[..., 1])).astype(np.float32)
    f0[~active] = 0.
    maps = dict(rho=rho, ff=ff, nmidb=nmidb, ddb=ddb,
                ndb=nmidb + ddb, t2star_ms=t2star, f0_rad=f0,
                b0_hz=angular_radial_b0_hz(fov, sc.b0_peak_hz))
    meta = dict(active_mask=active, radius_px=radius.astype(np.float32),
                phase_fraction=phases, canonical_xy=fov)
    return gt, maps, meta
