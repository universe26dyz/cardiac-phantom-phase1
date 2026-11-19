"""Continuous, static ventricular geometry; all world and local coordinates are mm."""
from dataclasses import dataclass, field
import numpy as np

DEFAULT_TISSUES = {0: (np.nan, np.nan, 0.0), 1: (1200., 45., 1.0),
                   2: (1600., 180., 1.0), 3: (1600., 180., 1.0)}
NAMES = {0: 'background', 1: 'LV myocardium', 2: 'LV blood', 3: 'RV blood'}

@dataclass
class Phantom:
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tissues: dict = field(default_factory=lambda: DEFAULT_TISSUES.copy())
    seed: int = 1234
    amplitude: float = 0.05
    wavelength_mm: float = 20.0

    def __post_init__(self):
        self.rotation = np.asarray(self.rotation, dtype=float)
        self.translation = np.asarray(self.translation, dtype=float)
        if (self.rotation.shape != (3, 3) or self.translation.shape != (3,) or
            not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-8) or
            not np.isclose(np.linalg.det(self.rotation), 1)):
            raise ValueError('rotation must be proper orthonormal and translation shape (3,)')
        if not 0 <= self.amplitude <= .05 or self.wavelength_mm <= 0:
            raise ValueError('amplitude must be in [0,0.05] and wavelength positive')
        if set(self.tissues) != {0, 1, 2, 3}:
            raise ValueError('tissues must have IDs 0 to 3')
        rng = np.random.default_rng(self.seed)
        directions = rng.normal(size=(3, 2, 8, 3))
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        self.wavevectors = directions * (2*np.pi / self.wavelength_mm)
        self.phases = rng.uniform(0, 2*np.pi, size=(3, 2, 8))
        # Fixed, export-independent integration points; bounding-box Monte Carlo.
        points = np.random.default_rng(91236).uniform([-33,-29,-56], [59,29,56], (30000,3))
        f = self.fields(self.local_to_world(points))
        labels = np.where(f['lv_inner'] <= 0, 2, np.where(f['lv_outer'] <= 0, 1, np.where(f['rv'] < 0, 3, 0)))
        self.centers = np.zeros((3,2))
        for k in (1,2,3):
            for j in range(2):
                self.centers[k-1,j] = self._raw_texture(points[labels == k], k, j).mean()

    def local_to_world(self, xyz):
        return np.asarray(xyz) @ self.rotation.T + self.translation

    def world_to_local(self, xyz):
        return (np.asarray(xyz) - self.translation) @ self.rotation

    def fields(self, xyz_mm):
        p = self.world_to_local(xyz_mm)
        if p.shape[-1:] != (3,):
            raise ValueError('coordinates must have shape (..., 3)')
        x, y, z = np.moveaxis(p, -1, 0)
        outer = (x/32)**2 + (y/28)**2 + (z/55)**2 - 1
        inner = (x/22)**2 + (y/18)**2 + ((z-2)/45)**2 - 1
        rv_ellipsoid = ((x-29)/29)**2 + (y/25)**2 + ((z-1)/43)**2 - 1
        # The shared epicardium is assigned to LV; RV is strictly outside.
        rv = np.maximum(rv_ellipsoid, -outer)
        return {'lv_outer': outer, 'lv_inner': inner, 'rv': rv}

    def _raw_texture(self, local, k, j):
        return np.sin(np.asarray(local) @ self.wavevectors[k-1,j].T + self.phases[k-1,j]).mean(axis=-1)

    def query(self, xyz_mm):
        f = self.fields(xyz_mm)
        inner, outer, rv = f['lv_inner'] <= 0, f['lv_outer'] <= 0, f['rv'] < 0
        label = np.zeros(np.shape(inner), dtype=np.uint8)
        label = np.where(outer & ~inner, 1, label)
        label = np.where(inner, 2, label)
        label = np.where(rv, 3, label).astype(np.uint8)
        result = {'label': label, 'valid': label != 0}
        local = self.world_to_local(xyz_mm)
        for i, name in enumerate(('t1_ms', 't2_ms', 'm0')):
            values = np.array(np.take([self.tissues[k][i] for k in range(4)], label), dtype=float, copy=True)
            if i < 2 and self.amplitude:
                for k in (1,2,3):
                    mask = label == k
                    if np.any(mask):
                        center = self.centers[k-1,i]
                        u = (self._raw_texture(local[mask], k, i) - center) / (1 + abs(center))
                        values[mask] *= 1 + self.amplitude*u
            result[name] = values
        return result


def sample_plane(phantom, center, column, row, spacing, shape):
    center, column, row = (np.asarray(v, dtype=float) for v in (center, column, row))
    if any(v.shape != (3,) or not np.all(np.isfinite(v)) for v in (center, column, row)):
        raise ValueError('center and basis must be finite 3-vectors')
    if not (np.isclose(np.linalg.norm(column), 1) and np.isclose(np.linalg.norm(row), 1) and np.isclose(column @ row, 0)):
        raise ValueError('plane basis vectors must be orthonormal')
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (2,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0) or len(shape) != 2 or any(int(n) != n or n <= 0 for n in shape):
        raise ValueError('spacing and shape must be positive row/column pairs')
    r = (np.arange(shape[0]) - (shape[0]-1)/2)*spacing[0]
    c = (np.arange(shape[1]) - (shape[1]-1)/2)*spacing[1]
    xyz = center + r[:, None, None]*row + c[None, :, None]*column
    return xyz, phantom.query(xyz)
