"""Synthetic cubes with known noise, for validating hi_cube_qa estimators."""

import numpy as np


def make_cube(nchan=24, ny=96, nx=96, seed=0, sigma=None, offsets=None, dtype=np.float64):
    """Gaussian-noise cube. Returns (cube, true_sigma[nchan], true_offset[nchan])."""
    rng = np.random.default_rng(seed)
    if sigma is None:
        sigma = 1e-3 * (1.0 + 0.5 * np.sin(np.linspace(0.0, 3.0, nchan)))
    if offsets is None:
        offsets = np.zeros(nchan)
    sigma = np.asarray(sigma, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    cube = rng.normal(size=(nchan, ny, nx)) * sigma[:, None, None] + offsets[:, None, None]
    return cube.astype(dtype), sigma, offsets
