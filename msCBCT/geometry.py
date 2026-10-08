"""geometry.py — NAF ray geometry built EXACTLY from the ASTRA cone_vec
vectors of the scanner.

The chain:

  SystemParams / BinnedGeometry  (params.py — scanner calibration)
        |
        v
  cone_vec 12-vectors per frame            <- identical formulas to
  (src | det centre d | u-step | v-step)      ../reconstruction/astra_recon.py
        |                                     ConeBeamReconstruction.vectors()
        v
  per-pixel ray:  origin = src,  through pixel centre
      pixel(r, c) = d + (c - (cols-1)/2) * u + (r - (rows-1)/2) * v
  (ASTRA convention, verified in astra-toolbox source:
   ConeVecProjectionGeometry3D.cpp converts the centre d to the corner
   detS = d - 0.5*rows*v - 0.5*cols*u, and the CUDA kernels sample pixel
   (r, c) at detS + (c+0.5)*u + (r+0.5)*v  — which equals the formula above.)

Key property exploited for NAF sampling (verified numerically to <= 1e-13
voxels): the frame-t vectors are EXACTLY the z-rotation of the frame-0 vectors,

    vectors(t) = Rz(rot_dir * t) @ vectors(0)     (all four 3-vectors),

so per-pixel sample points are computed ONCE per source at gantry 0 deg and
each training ray is then rotated to its own frame angle — the same
"compute at 0 deg, then rotate" scheme the original NAF code used, but with
the exact ASTRA rays.

Coordinate conventions (all inherited from the ASTRA pipeline; no empirical
index flips anywhere):

  * Geometry frame = ASTRA volume frame: origin at the rotation axis /
    volume centre, units = 1 voxel = 1 binned detector pixel (pixel_size mm).
  * Volume voxel (ix, iy, iz) centre sits at
        ( ix + 0.5 - nx/2,  iy + 0.5 - ny/2,  iz + 0.5 - nz/2 )   [voxels]
    with the python/ASTRA volume array indexed [iz, iy, ix]
    (verified in astra cuda/3d/cone_bp.cu: fX = X - 0.5*iVolX + 0.5).
  * Projection frame k: band row r increases towards +z (preprocessing
    v-flips the bands), column c increases along ASTRA's u vector, which at
    gantry 0 deg is +x.  The NIfTI on disk is (H, cols, N); SimpleITK's numpy
    view is therefore (N, cols, H) = (frame, c, r).
  * NAF normalized coordinates: divide the voxel-frame coordinates by
    R = SOD / pixel_size  (so [-1, 1] spans ±SOD, source orbit radius = 1.0,
    like the original NAF code — but here R is derived, never hand-configured).
"""

import numpy as np

from params import BinnedGeometry


# =============================================================================
# 1. Frame schedule: per-frame gantry angle + firing source
# =============================================================================

def frame_schedule(num_proj, geom: BinnedGeometry, orbit_deg=360.0,
                   intra_view_stepping=True):
    """Gantry angle [deg] and source index of every frame of the scan.

    The stored projection file holds ``num_proj = num_views * num_src`` frames,
    frame k belonging to view k // num_src and source k % num_src (firing
    order, verified by the preprocessing phase check).  Views are spaced
    ``orbit_deg / num_views`` apart.

    intra_view_stepping:
      True  — real scanner: the gantry rotates CONTINUOUSLY, deg_per_frame
              (0.125 deg) per frame, so source s of a view fires
              s * deg_per_frame LATER than the view's nominal angle.  This
              matches angle(k) = k * 0.125 deg for the full scan, and remains
              exact for the saved evenly-spaced view subsets (each kept view
              is a complete 8-frame cycle).
      False — simulated data where all sources fire at the view angle.

    Returns (frame_angle_deg (num_proj,) float64, frame_source (num_proj,) int).
    """
    ns = geom.params.source_number
    if num_proj % ns:
        raise ValueError(f"num_proj={num_proj} is not a multiple of num_src={ns}")
    num_views = num_proj // ns
    k = np.arange(num_proj)
    view, src = k // ns, k % ns
    ang = view * (orbit_deg / num_views)
    if intra_view_stepping:
        ang = ang + src * geom.params.deg_per_frame
    return ang.astype(np.float64), src.astype(np.int64)


# =============================================================================
# 2. cone_vec vectors (identical math to the SIRT reference)
# =============================================================================

def cone_vec_vectors(geom: BinnedGeometry, frame_angle_deg, frame_source,
                     rot_dir=+1):
    """The (N, 12) ASTRA cone_vec array in voxel units (voxel = pixel width).

    Row k = [srcX srcY srcZ, dX dY dZ, uX uY uZ, vX vY vZ] for frame k.
    The same construction as ../reconstruction/astra_recon.py (the SIRT
    reference).
    """
    P, vx = geom.params, geom.pixel_size
    t = np.radians(np.asarray(frame_angle_deg, np.float64)) * rot_dir
    src_z = geom.z_src[frame_source] / vx
    det_z = geom.z_det[frame_source] / vx
    sod, odd = P.SOD / vx, P.ODD / vx
    off_u, off_v = geom.offset_u / vx, geom.offset_v / vx
    pitch = 1.0                                     # detector pixel = 1 voxel

    V = np.zeros((t.size, 12))
    V[:, 0], V[:, 1], V[:, 2] = np.sin(t) * sod, -np.cos(t) * sod, src_z
    V[:, 3] = -np.sin(t) * odd + np.cos(t) * off_u
    V[:, 4] = np.cos(t) * odd + np.sin(t) * off_u
    V[:, 5] = det_z + off_v
    V[:, 6], V[:, 7] = np.cos(t) * pitch, np.sin(t) * pitch   # u: horizontal
    V[:, 11] = pitch                                          # v: along +z
    return V


def base_vectors(geom: BinnedGeometry):
    """(num_src, 12) cone_vec vectors at gantry angle 0 (one row per source)."""
    ns = geom.params.source_number
    zeros = np.zeros(ns)
    return cone_vec_vectors(geom, zeros, np.arange(ns), rot_dir=+1)


# =============================================================================
# 3. Per-pixel rays at gantry 0 deg
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


def rays_zero_deg(geom: BinnedGeometry):
    """Exact per-source rays at gantry 0 deg, in voxel units.

    Returns
    -------
    origins : (num_src, 3) float64 — source positions (0, -SOD_vx, z_src_vx).
    dirs    : (num_src, H * cols, 3) float32 — UNIT direction source -> pixel
              centre, flattened detector index = r * cols + c (row-major,
              r = band row increasing +z, c = column increasing +x at 0 deg).
    valid   : (num_src, H * cols) bool — True where the band row is covered by
              the physical panel; rows outside carry no measurement (the
              preprocessing wrote zeros there).  ``geom.data_j`` gives that
              range on the RAW (unflipped) frame, so it is MIRRORED here —
              see the comment in the loop.
    """
    ns, H, cols = geom.params.source_number, geom.H, geom.cols
    base = base_vectors(geom)

    origins = base[:, 0:3].copy()
    dirs = np.empty((ns, H * cols, 3), dtype=np.float32)
    valid = np.zeros((ns, H * cols), dtype=bool)
    for s in range(ns):
        pix = pixel_centers(base[s], H, cols)             # (H, cols, 3)
        d = pix - origins[s][None, None, :]
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        dirs[s] = d.reshape(-1, 3).astype(np.float32)
        # data_j is a RAW-frame row range: the MATLAB tables (data_rows) and
        # params.band_tables_raw live on the unflipped frame, and the
        # preprocessing zeroes those rows and THEN flips the band
        #   msCBCT_preprocessing_ASRS.m:  p(...) = 0;  flipud(p)
        # so in the stored band (row increasing with +z) the measured rows are
        # the mirrored range.  Only sources 1 and 8 are partial:
        #   s1 raw [0,43)  -> stored [22,65)      s8 raw [5,65) -> stored [0,60)
        j0, j1 = geom.data_j[s]
        m = np.zeros((H, cols), dtype=bool)
        m[H - j1:H - j0, :] = True
        valid[s] = m.reshape(-1)
    return origins, dirs, valid


# =============================================================================
# 4. Gantry rotation with PER-RAY angles
# =============================================================================

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


# =============================================================================
# 5. Ray sampling (equal-length rays, 1-voxel steps, like the original NAF code)
# =============================================================================

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
