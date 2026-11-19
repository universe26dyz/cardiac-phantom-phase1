"""Integrate signal samples, including background, and add observation noise."""
import numpy as np

def integrate(samples, weights):
    samples = np.asarray(samples, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if samples.ndim < 1 or weights.ndim != 1 or samples.shape[0] != weights.size:
        raise ValueError('sample axis and PSF weights must agree')
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError('PSF weights must be nonnegative and sum to one')
    return np.tensordot(weights, samples, axes=(0, 0))

def add_noise(clean, sigma, seed):
    clean = np.asarray(clean)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError('sigma must be finite and nonnegative')
    if sigma == 0:
        return clean.copy()
    return clean + np.random.default_rng(seed).normal(0, sigma, clean.shape)
