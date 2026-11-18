"""V10.10 synthetic MRI water + nine-fat-peak echo signal, NumPy-only."""
import numpy as np

FAT9_PPM_OFFSETS = np.array([-3.80, -3.40, -3.10, -2.68, -2.46,
                             -1.95, -0.50, 0.49, 0.59], np.float32)


def angular_radial_b0_hz(coords_xy, peak_hz=150.):
    """Original V10.3 B0: quadrant-wise angular ramp times cos(2*pi*r)."""
    coords = np.asarray(coords_xy, np.float32)
    x, y = coords[..., 0], coords[..., 1]
    th = np.arctan2(y, x)
    halfpi = np.pi / 2.
    p = float(peak_hz)
    angular = np.where(th >= 0.,
                       np.where(th <= halfpi, p * th / halfpi,
                                p * (1. - (th - halfpi) / halfpi)),
                       np.where(th >= -halfpi, p * th / halfpi,
                                -p * (1. - ((-th) - halfpi) / halfpi)))
    angular = np.where(x*x + y*y < 1e-14, 0., angular)
    r = np.sqrt(x*x + y*y)
    return (angular * np.cos(2*np.pi*r)).astype(np.float32)


def spectral_signal(rho, ff, nmidb, ddb, f0_rad, t2star_ms,
                    b0_hz, te_ms, larmor_hz_per_ppm=128.):
    """Complex signal per material point at TE (ms); identical to V10.3 model.

    Water + nine-peak fat spectrum, exponential T2* and B0 phase accumulation;
    B0 is evaluated at the *deformed* position at the acquisition event.
    """
    rho, ff = np.asarray(rho, np.float32), np.asarray(ff, np.float32)
    nmidb, ddb = np.asarray(nmidb, np.float32), np.asarray(ddb, np.float32)
    ndb = nmidb + ddb
    weights = np.stack([
        np.full_like(ndb, 9.), 76.8 - 6.5*ndb + 2.*nmidb,
        np.full_like(ndb, 6.), 4.*ddb, np.full_like(ndb, 6.),
        2.*nmidb, np.full_like(ndb, 4.), np.full_like(ndb, 1.),
        2.*ndb], axis=-1).astype(np.float32)
    weights /= np.sum(weights, axis=-1, keepdims=True)
    te = np.asarray(te_ms, np.float32)
    te_s = te * 1e-3
    offsets_hz = FAT9_PPM_OFFSETS * float(larmor_hz_per_ppm)
    chem = np.sum(weights * np.exp(1j*2*np.pi*te_s[..., None]*offsets_hz), axis=-1)
    mix = (1. - ff) + ff*chem
    t2 = np.asarray(t2star_ms, np.float32)
    decay = np.exp(-te/np.where(t2 > 0., t2, 1.))
    phase = np.exp(1j*(np.asarray(f0_rad, np.float32) +
                        2*np.pi*np.asarray(b0_hz, np.float32)*te_s))
    return (rho*decay*phase*mix).astype(np.complex64)
