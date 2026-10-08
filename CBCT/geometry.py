"""geometry.py — SINGLE-SOURCE cone-beam geometry on the exact ASTRA cone_vec
convention (the same convention as ../msCBCT).

Two acquisition types are supported and are told apart from the projection
FRAME SIZE alone — there is no mode switch in the config:

    frame (cols, rows)          layout       source z        detector centre z
    ------------------------------------------------------------------------
    (geom.cols, geom.rows)      'panel'      0 (on axis)     cbct_offset_v
    (geom.cols, geom.H)         'band'       geom.z_src[s]   geom.z_det[s]

'panel' = a conventional CBCT scan of this scanner: one source on the central
axis, the full flat panel, half-fan (offset_u) in u and the panel's own
vertical offset (``SystemParams.cbct_offset_v``) in v.

'band' = ONE source of the msCBCT scan, i.e. the 65-row band of source s
extracted from the msCBCT projection file.  Its source sits off-axis at
z_src[s], its band is centred at z_det[s], and — because the real gantry
rotates continuously — source s fires deg_per_frame * s LATER than the view's
nominal angle.  All of that follows from the source index, which is the ONE
thing the frame size cannot tell us (all 8 bands are 65 rows), so it comes
from the config key ``source``.

Fan and cone angles are never read from sensor-position files: they follow
analytically from (rows, cols, pixel_size, SDD, offset_u, offset_v, z_src,
z_det) via the ASTRA pixel formula, which is exactly "derive the geometry
from the projection size".

v ORDER
-------
Rows are expected in ASTRA order (row index increases with +z), which is what
the preprocessing writes for both panels and bands.  ``flip_v: true`` reads
files stored the other way round (row 0 at +z).  Only the DATA is reordered;
the geometry stays pure ASTRA.  A z flip of an on-axis panel is a symmetry of
its own data, so check the orientation against an independent reconstruction
(``../reconstruction/CBCT_reconstruction.ipynb``).
"""

import numpy as np

from params import BinnedGeometry


# =============================================================================
# 1. Layout detection
# =============================================================================

class ScanLayout:
    """Everything about the acquisition that the frame size (+ source) fixes."""

    def __init__(self, kind, rows, cols, z_src_mm, z_det_mm, source,
                 flip_v, angle_step_extra_deg, valid):
        self.kind = kind                          # 'panel' | 'band'
        self.rows, self.cols = rows, cols
        self.z_src_mm, self.z_det_mm = z_src_mm, z_det_mm
        self.source = source                      # 0-based, None for 'panel'
        self.flip_v = flip_v
        self.angle_step_extra_deg = angle_step_extra_deg   # intra-view offset
        self.valid = valid                        # (rows, cols) bool
        self.num_det = rows * cols

    def summary(self):
        src = "on axis" if self.kind == "panel" else f"msCBCT source {self.source + 1}"
        print(f"Layout        : '{self.kind}' — {src}, frame {self.cols} x "
              f"{self.rows} (cols x rows)")
        print(f"  z_src / z_det : {self.z_src_mm:+.2f} / {self.z_det_mm:+.2f} mm")
        print(f"  v order       : row 0 -> {'-z (ASTRA, no flip)' if not self.flip_v else '+z in file -> FLIPPED on load'}")
        print(f"  angle offset  : {self.angle_step_extra_deg:+.3f} deg "
              f"({'continuous-rotation firing delay' if self.angle_step_extra_deg else 'none'})")
        n_bad = self.valid.size - int(self.valid.sum())
        print(f"  valid pixels  : {int(self.valid.sum())} / {self.valid.size}"
              + (f" ({n_bad} never-measured masked)" if n_bad else ""))


def detect_layout(frame_shape, geom: BinnedGeometry, source=None, flip_v=False,
                  is_mscbct=None):
    """Identify the acquisition from the projection FRAME shape.

    frame_shape : (cols, rows) — proj.shape[1:] of the SimpleITK numpy view.
    source      : 1-BASED msCBCT source index; required for a band, ignored
                  for a full panel.
    flip_v      : True = file row 0 is the +z end (flip on load).
    is_mscbct   : config ``msCBCT``. None = trust the frame size. True/False
                  DECLARES which acquisition this is and is cross-checked
                  against the frame size, so a config pointed at the wrong
                  file fails loudly instead of silently reconstructing with
                  the other geometry.
    """
    cols, rows = int(frame_shape[0]), int(frame_shape[1])
    if cols != geom.cols:
        raise ValueError(
            f"projection has {cols} columns, this scanner's binned panel has "
            f"{geom.cols} — wrong voxel_size for this file?")

    if is_mscbct is not None:
        declared = "band" if is_mscbct else "panel"
        found = ("panel" if rows == geom.rows else
                 "band" if rows == geom.H else None)
        if found is not None and found != declared:
            if declared == "band":
                raise ValueError(
                    f'config says "msCBCT": true (one source of the '
                    f"multi-source scan) but the frame is {cols} x {rows} = "
                    f"the FULL flat panel, i.e. a conventional CBCT scan. Set "
                    f'"msCBCT": false, or point proj_file at msCBCT data.')
            raise ValueError(
                f'config says "msCBCT": false (conventional CBCT) but the '
                f"frame is {cols} x {rows} = ONE msCBCT band. Set "
                f'"msCBCT": true and "source": 1..{geom.params.source_number}.')
        print(f'[layout] config "msCBCT": {bool(is_mscbct)} — confirmed by the '
              f"frame size ({cols} x {rows})")

    if rows == geom.rows:                                   # ---- full panel
        if source is not None:
            print(f"[layout] full-panel frame ({rows} rows): config 'source' "
                  f"= {source} ignored (on-axis source)")
        valid = np.ones((rows, cols), dtype=bool)
        return ScanLayout("panel", rows, cols, 0.0, geom.params.cbct_offset_v,
                          None, bool(flip_v), 0.0, valid)

    if rows == geom.H:                                      # ---- msCBCT band
        if source is None:
            raise ValueError(
                f"frame has {rows} rows = one msCBCT band, but all "
                f"{geom.params.source_number} bands are {geom.H} rows so the "
                f"size cannot say WHICH source it is. Set "
                f'"source": <1..{geom.params.source_number}> in the config '
                f'(file section, next to "msCBCT": true).')
        s = int(source) - 1
        if not 0 <= s < geom.params.source_number:
            raise ValueError(f"source must be 1..{geom.params.source_number}, "
                             f"got {source}")
        # data_j is a RAW-frame range; the preprocessing zeroes then flips the
        # band, so the measured stored rows are the mirrored range
        # (s1 [0,43) -> [22,65), s8 [5,65) -> [0,60) at 0.4 mm).
        j0, j1 = geom.data_j[s]
        valid = np.zeros((rows, cols), dtype=bool)
        valid[rows - j1:rows - j0, :] = True
        return ScanLayout("band", rows, cols, float(geom.z_src[s]),
                          float(geom.z_det[s]), s, bool(flip_v),
                          s * geom.params.deg_per_frame, valid)

    raise ValueError(
        f"frame has {rows} rows; expected {geom.rows} (full panel) or "
        f"{geom.H} (one msCBCT band) at voxel_size {geom.pixel_size / 10:.3f} cm")


# =============================================================================
# 2. cone_vec vector at gantry 0
# =============================================================================

def base_vector(layout: ScanLayout, geom: BinnedGeometry):
    """The 12-vector [src | det centre | u step | v step] at gantry 0, voxel units.

    Same formulas as ../msCBCT/geometry.py cone_vec_vectors and
    ../reconstruction/astra_recon.py, with the per-source z tables replaced by
    this layout's scalars.
    """
    P, vx = geom.params, geom.pixel_size
    sod, odd = P.SOD / vx, P.ODD / vx
    off_u, off_v = geom.offset_u / vx, geom.offset_v / vx
    v12 = np.zeros(12)
    v12[0], v12[1], v12[2] = 0.0, -sod, layout.z_src_mm / vx      # source
    v12[3], v12[4] = off_u, odd                                   # det centre
    v12[5] = layout.z_det_mm / vx + off_v
    v12[6] = 1.0                                                  # u = +x
    v12[11] = 1.0                                                 # v = +z
    return v12


# =============================================================================
# 3. Per-pixel rays at gantry 0
# =============================================================================

def rays_zero_deg(layout: ScanLayout, geom: BinnedGeometry):
    """Exact rays at gantry 0, voxel units.

    Returns
    -------
    origin : (3,) float64     — source position.
    dirs   : (num_det, 3) float32 — unit direction source -> pixel centre,
             flattened index r * cols + c (r increases with +z, c with +x).
    valid  : (num_det,) bool  — flattened layout.valid.
    """
    v12 = base_vector(layout, geom)
    origin = v12[0:3].copy()
    pix = pixel_centers(v12, layout.rows, layout.cols)         # (rows, cols, 3)
    d = pix - origin[None, None, :]
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return origin, d.reshape(-1, 3).astype(np.float32), layout.valid.reshape(-1)


# =============================================================================
# 4. View schedule
# =============================================================================

def z_coverage_mm(layout: ScanLayout, geom: BinnedGeometry):
    """(z_min, z_max) [mm] that this single source actually measures at the
    rotation axis: the measured detector rows' z extent projected back to
    y = 0.

    A single source measures far less z than the reconstruction FOV usually
    spans — a full panel ~76 mm, one msCBCT band only ~17 mm — and voxels
    outside it are unconstrained by the data.
    """
    P, vx = geom.params, geom.pixel_size
    r = np.flatnonzero(layout.valid.any(axis=1))                  # measured rows
    z_pix = (layout.z_det_mm + geom.offset_v
             + (np.array([r[0], r[-1]]) - (layout.rows - 1) / 2.0) * vx)
    z_axis = layout.z_src_mm + (z_pix - layout.z_src_mm) * (P.SOD / P.SDD)
    return float(z_axis[0]), float(z_axis[1])


def deinterleave_if_needed(proj, source, num_src):
    """Keep only ``source``'s frames if ``proj`` is the FULL msCBCT file.

    The full file stores frame k = view * num_src + src, so its per-frame mean
    level JUMPS between neighbouring frames (different sources, very different
    z through the object) while staying smooth at lag num_src (same source,
    one view apart).  A file that already holds a single source is smooth at
    both lags.  That ratio identifies the file, so the user can point the
    config at either one without a new option.

    Returns (proj, was_interleaved).
    """
    n = proj.shape[0]
    if n % num_src or n < 4 * num_src:
        return proj, False
    lvl = np.abs(proj).mean(axis=(1, 2))
    d1 = np.abs(np.diff(lvl)).mean()
    dn = np.abs(lvl[num_src:] - lvl[:-num_src]).mean()
    if d1 > 3.0 * max(dn, 1e-12):
        return proj[source::num_src], True
    return proj, False


def view_angles(num_views, layout: ScanLayout, orbit_deg=360.0):
    """Gantry angle [deg] of every stored view.

    Views are ``orbit_deg / num_views`` apart.  A band adds this source's
    firing delay (source s fires s * deg_per_frame after the view's nominal
    angle — the real gantry never stops rotating); a panel scan adds nothing.
    """
    return (np.arange(num_views) * (orbit_deg / num_views)
            + layout.angle_step_extra_deg).astype(np.float64)


# =============================================================================
# 5. Ray helpers (identical to ../msCBCT/geometry.py)
# =============================================================================

def pixel_centers(vec12, rows, cols):
    """Pixel-centre positions (rows, cols, 3) for ONE cone_vec row.

    ASTRA convention: pixel (r, c) centre = d + (c - (cols-1)/2) u
                                          + (r - (rows-1)/2) v.
    """
    d, u, v = vec12[3:6], vec12[6:9], vec12[9:12]
    cu = (np.arange(cols) - (cols - 1) / 2.0)[None, :, None]   # (1, C, 1)
    cv = (np.arange(rows) - (rows - 1) / 2.0)[:, None, None]   # (R, 1, 1)
    return d[None, None, :] + cu * u[None, None, :] + cv * v[None, None, :]


def rotate_z(xyz, angle_deg, rot_dir=+1):
    """Rotate points about the global z axis, one angle PER LEADING ENTRY.

    xyz       : (n, ..., 3) array (any dtype); rotated in place semantics-free
                (a new array is returned).
    angle_deg : scalar or (n,) — frame angle of each leading entry.
    Matches the ASTRA orbit: vectors(t) = Rz(rot_dir * t) @ vectors(0).
    """
    a = np.deg2rad(np.asarray(angle_deg, dtype=np.float64)) * rot_dir
    cos_a, sin_a = np.cos(a), np.sin(a)
    # broadcast (n,) over (n, ..., 3)
    extra = xyz.ndim - 1 - np.ndim(cos_a)
    shape = np.shape(cos_a) + (1,) * extra
    cos_a = np.reshape(cos_a, shape).astype(xyz.dtype, copy=False)
    sin_a = np.reshape(sin_a, shape).astype(xyz.dtype, copy=False)

    out = np.empty_like(xyz)
    out[..., 0] = cos_a * xyz[..., 0] - sin_a * xyz[..., 1]
    out[..., 1] = sin_a * xyz[..., 0] + cos_a * xyz[..., 1]
    out[..., 2] = xyz[..., 2]
    return out


def sample_offsets(num_samples):
    """Distances from the source [normalized units, 1.0 = SOD], span [0, 2]."""
    return (np.linspace(-1, 1, int(num_samples), dtype=np.float32) + 1.0)


# =============================================================================
# 6. Reconstruction grid = EXACT ASTRA voxel centres
# =============================================================================

def volume_shape(img_size_mm, voxel_size_mm):
    """(nx, ny, nz) voxel counts, same rounding as the ASTRA pipeline."""
    return tuple(int(round(s / voxel_size_mm)) for s in img_size_mm)


def astra_grid_chunks(nx, ny, nz, R_norm, chunk_z=16):
    """Yield normalized (x, y, z) coordinates of ASTRA voxel centres, chunked
    along z.  Yields (xyz (nc*ny*nx, 3) float32, z0, z1) with the point order
    matching a volume array indexed [iz, iy, ix].
    """
    x = ((np.arange(nx) + 0.5 - nx / 2.0) / R_norm).astype(np.float32)
    y = ((np.arange(ny) + 0.5 - ny / 2.0) / R_norm).astype(np.float32)
    z = ((np.arange(nz) + 0.5 - nz / 2.0) / R_norm).astype(np.float32)
    for z0 in range(0, nz, chunk_z):
        z1 = min(z0 + chunk_z, nz)
        zg, yg, xg = np.meshgrid(z[z0:z1], y, x, indexing="ij")
        xyz = np.stack([xg, yg, zg], axis=-1).reshape(-1, 3)
        yield xyz, z0, z1
