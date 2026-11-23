"""Small synthetic cubes and CARACal-like session trees for the tests."""
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

FREQ0_HZ = 1.40e9
DF_HZ = 26367.19


def cube_header(nchan, ny, nx, bmaj_deg=20.0 / 3600, bmin_deg=15.0 / 3600, bpa=10.0):
    hdr = fits.Header()
    hdr["CTYPE1"], hdr["CRVAL1"], hdr["CDELT1"], hdr["CRPIX1"] = "RA---SIN", 150.0, -6.0 / 3600, nx / 2 + 1
    hdr["CTYPE2"], hdr["CRVAL2"], hdr["CDELT2"], hdr["CRPIX2"] = "DEC--SIN", 2.0, 6.0 / 3600, ny / 2 + 1
    hdr["CTYPE3"], hdr["CRVAL3"], hdr["CDELT3"], hdr["CRPIX3"] = "FREQ", FREQ0_HZ, DF_HZ, 1
    hdr["BMAJ"], hdr["BMIN"], hdr["BPA"] = bmaj_deg, bmin_deg, bpa
    hdr["BUNIT"] = "JY/BEAM"
    return hdr


def write_cube(path, nchan=16, ny=48, nx=48, sigma=1e-3, seed=0, source=None, header_extra=None):
    """A Gaussian-noise cube; *source* = (channel, y, x, amplitude) adds one bright pixel."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, sigma, (nchan, ny, nx)).astype(np.float32)
    if source is not None:
        c, y, x, amp = source
        data[c, y, x] += amp
    hdr = cube_header(nchan, ny, nx)
    for k, v in (header_extra or {}).items():
        hdr[k] = v
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)
    return data


def make_session(root, obsid="1600000000_sdp_l0", pointings=("p01", "p02"), products=("dirty", "image", "residual"),
                 n_ant=8, flag_frac=0.2, **cube_kw):
    """Build <root>/<obsid>/caracal/ with cubes, a flag report and an obsinfo summary; return the caracal path."""
    car = Path(root) / obsid / "caracal"
    for i, p in enumerate(pointings):
        cdir = car / f"output_hi_{p}" / "cubes" / "cube_1"
        cdir.mkdir(parents=True)
        for j, prod in enumerate(products):
            write_cube(cdir / f"G23_{obsid[:10]}_{p}_HI_1.{prod}.fits", seed=10 * i + j, **cube_kw)
    diag = car / "output_hi_1gc" / "diagnostic_plots"
    diag.mkdir(parents=True)
    report = {"Flag stats": [{"corrs": {"0": {"name": "XX", "frac": flag_frac, "sum": flag_frac * 1e6, "counts": 1e6},
                                       "1": {"name": "YY", "frac": flag_frac, "sum": flag_frac * 1e6, "counts": 1e6}}}]}
    (diag / f"{obsid}-cal-flag__bpcal-flagging-summary-0.json").write_text(json.dumps(report))
    info = car / "output_hi_1gc" / "obsinfo"
    info.mkdir(parents=True)
    (info / f"{obsid[:10]}-obsinfo-summary.json").write_text(json.dumps({"ANT": {"NAME": [f"m{k:03d}" for k in range(n_ant)], "DISH_DIAMETER": [13.5] * n_ant}}))
    return car
