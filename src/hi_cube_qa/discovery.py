"""Walk a CARACal session tree and group cubes for QA.

Directory layout produced by the CARACal HI worker:

   <session>_sdp_l0/caracal/
       output_hi_<pXX>/                 <- one imaging run per pointing
           cubes/
               cube_1/  G23_<obsid>_<pXX>_HI_1.<product>.fits
               cube_2/  G23_<obsid>_<pXX>_HI.<product>.fits
       output_hi_1gc/diagnostic_plots/  <- flagging-summary JSONs

<product> is one of: dirty, image, residual, model, psf, ...

A QA "unit" is one (pointing, cube_N) pair. All <product> FITS inside the
same cube_N share the spatial WCS *and* channel grid, so they can be safely
overlaid on the same axes. Different cube_N (different spectral windows) are
kept as separate units.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Products that can be plotted; used as the default selection and to order the
# legend. Anything else discovered (e.g. "psf") is still available via
# products: all.
KNOWN_PRODUCTS = ("dirty", "image", "residual", "line", "model", "psf")

_POINTING_RE = re.compile(r"^output_hi_(p\d+[a-z0-9_]*)$")
_CUBE_RE = re.compile(r"^cube_(\w+)$")


def _norm_cube(s) -> str:
    """Normalise a cube identifier to its bare suffix so that 'cube_1', 'c1' and
    '1' all compare equal (used for the optional cube filter).
    """
    s = str(s).strip().lower()
    if s.startswith("cube_"):
        return s[5:]
    if s.startswith("cube"):
        return s[4:]
    if s.startswith("c"):
        return s[1:]
    return s


@dataclass
class CubeProduct:
    """A single <product>.fits file inside one cube_N directory."""

    path: Path
    product: str  # 'dirty' | 'image' | 'residual' | ...
    pointing: str  # 'p63'
    cube_id: str  # 'cube_1'
    label: str  # 'c1-image'  (used in plots / legends)


@dataclass
class Unit:
    """One (pointing, cube_N) pair: products that share a WCS + channel grid."""

    pointing: str  # 'p63'
    cube_id: str  # 'cube_1'
    products: list[CubeProduct] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.pointing}/{self.cube_id}"

    @property
    def cube_short(self) -> str:
        # cube_1 -> c1
        m = _CUBE_RE.match(self.cube_id)
        return f"c{m.group(1)}" if m else self.cube_id


@dataclass
class Session:
    """One observation session (one caracal/ root)."""

    obsid: str  # '1610268336_sdp_l0'
    root: Path
    units: list[Unit] = field(default_factory=list)

    @property
    def pointings(self) -> list[str]:
        seen = []
        for u in self.units:
            if u.pointing not in seen:
                seen.append(u.pointing)
        return seen

    def units_for(self, pointing: str) -> list[Unit]:
        return [u for u in self.units if u.pointing == pointing]


def _parse_product(fits_path: Path) -> str | None:
    """Extract the imaging product from a cube filename.

        G23_..._HI.image.fits      -> 'image'
        G23_..._HI_1.residual.fits -> 'residual'
        imsub-line.fits            -> 'line' (continuum-subtracted cube)

    Returns None for names without an alphabetic product token (e.g. channel
    split images like ...-0000-image.fits are still fine; truly unparseable
    names are skipped).
    """
    name = fits_path.name
    if name == "imsub-line.fits":
        return "line"
    if not name.endswith(".fits"):
        return None
    stem = name[: -len(".fits")]
    token = stem.rsplit(".", 1)[-1].lower()
    # product tokens are alphabetic words; reject pure numbers / channel ids
    if token.isalpha():
        return token
    return None


def _pointing_sort_key(pointing: str) -> tuple:
    """Sort p7 before p13 before p13b (numeric-aware)."""
    m = re.match(r"p(\d+)(.*)", pointing)
    if m:
        return (int(m.group(1)), m.group(2))
    return (1 << 30, pointing)


def discover_session(
    root: str | Path,
    products: list[str] | str = "all",
    pointings: list[str] | None = None,
    cubes: list | str | None = None,
) -> Session:
    """Discover QA units under a single CARACal session root.

    Parameters
    ----------
    root      : path to the session's ``caracal/`` directory
    products  : list of product names to keep (e.g. ['dirty','image','residual'])
                or the string 'all' to keep every product found.
    pointings : optional list of pointing ids to keep (e.g. ['p63','p64']);
                None keeps all discovered pointings.
    cubes     : optional list of cube ids to keep. Accepts 'cube_1', 'c1' or 1
                (all equivalent). None or 'all' keeps every cube_N found.

    Returns
    -------
    Session with units sorted by pointing then cube id. Units with no surviving
    products (after filtering) are dropped.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Session root not found or not a directory: {root}")

    obsid = root.parent.name  # .../<obsid>_sdp_l0/caracal -> <obsid>_sdp_l0

    keep_all = isinstance(products, str) and products.lower() == "all"
    keep_set = None if keep_all else {p.lower() for p in products}
    point_set = None if pointings is None else set(pointings)
    cube_all = cubes is None or (isinstance(cubes, str) and cubes.lower() == "all")
    cube_set = None if cube_all else {_norm_cube(c) for c in cubes}

    units: dict[tuple[str, str], Unit] = {}

    for pdir in sorted(root.glob("output_hi_*")):
        if not pdir.is_dir():
            continue
        m = _POINTING_RE.match(pdir.name)
        if not m:
            continue  # skips output_hi_1gc, etc.
        pointing = m.group(1)
        if point_set is not None and pointing not in point_set:
            continue

        for cube_dir in sorted((pdir / "cubes").glob("cube_*")):
            if not cube_dir.is_dir():
                continue
            cm = _CUBE_RE.match(cube_dir.name)
            if not cm:
                continue
            if cube_set is not None and _norm_cube(cube_dir.name) not in cube_set:
                continue
            cube_id = cube_dir.name
            cube_short = f"c{cm.group(1)}"

            for fits_path in sorted(cube_dir.glob("*.fits")):
                product = _parse_product(fits_path)
                if product is None:
                    continue
                if keep_set is not None and product not in keep_set:
                    continue

                cp = CubeProduct(
                    path=fits_path,
                    product=product,
                    pointing=pointing,
                    cube_id=cube_id,
                    label=f"{cube_short}-{product}",
                )
                ukey = (pointing, cube_id)
                if ukey not in units:
                    units[ukey] = Unit(pointing=pointing, cube_id=cube_id)
                units[ukey].products.append(cp)

    # Order products within each unit by KNOWN_PRODUCTS, unknowns last (alpha).
    def _prod_rank(cp: CubeProduct) -> tuple:
        try:
            return (KNOWN_PRODUCTS.index(cp.product), cp.product)
        except ValueError:
            return (len(KNOWN_PRODUCTS), cp.product)

    unit_list = [u for u in units.values() if u.products]
    for u in unit_list:
        u.products.sort(key=_prod_rank)

    unit_list.sort(key=lambda u: (_pointing_sort_key(u.pointing), u.cube_id))

    return Session(obsid=obsid, root=root, units=unit_list)


# Fallback globs tried (in order) when the configured pattern finds nothing,
# so flag reports are picked up even if a session's layout differs slightly.
_FLAG_FALLBACKS = [
    "output_hi_1gc/diagnostic_plots/*-flagging-summary-*.json",
    "output_hi_*gc/diagnostic_plots/*-flagging-summary-*.json",
    "output_hi_*/diagnostic_plots/*-flagging-summary-*.json",
    "**/diagnostic_plots/*-flagging-summary-*.json",
    "**/*-flagging-summary-*.json",
]


def discover_flag_reports(
    root: str | Path,
    glob_pattern: str | list | None,
) -> tuple[list[Path], str | None]:
    """Resolve flagging-summary JSONs for a session, robustly.

    *glob_pattern* may be a single glob, a list of globs (tried in order), or
    None to disable. Whatever is given is tried first; if nothing matches, a
    series of increasingly broad fallback globs is tried so that a session whose
    layout differs slightly still has its flags found. Crosscal-summary and
    *-plots-*.html files are naturally excluded (the pattern requires
    '-flagging-summary-*.json').

    Returns (sorted_paths, pattern_that_matched) - ([], None) if disabled or
    nothing was found anywhere.
    """
    if not glob_pattern:
        return [], None
    root = Path(root)

    patterns: list[str] = [glob_pattern] if isinstance(glob_pattern, str) else list(glob_pattern)
    for fb in _FLAG_FALLBACKS:
        if fb not in patterns:
            patterns.append(fb)

    for pat in patterns:
        try:
            hits = sorted(p for p in root.glob(pat) if p.is_file())
        except (ValueError, OSError):
            continue
        if hits:
            return hits, pat
    return [], None


def summarise(session: Session) -> str:
    """One-line-per-unit human summary, for logging."""
    lines = [
        f"Session {session.obsid}: {len(session.units)} unit(s) across "
        f"{len(session.pointings)} pointing(s)"
    ]
    for u in session.units:
        prods = ", ".join(p.product for p in u.products)
        lines.append(f"  {u.key:<16} [{prods}]")
    return "\n".join(lines)
