"""reproject_parallel.py — geometry-DECOUPLED parallel-beam re-projection of a
trained NAF/INR with ASTRA.

WHY PARALLEL, AND WHY THIS IS THE WHOLE POINT
---------------------------------------------
A NAF network is a function  mu(x, y, z)  — a volume, not a scan.  The
acquisition geometry (8 sources, half-fan offset, 0.125 deg/frame, partial
bands, cone divergence) is exactly what the training removed from the data;
re-projecting through it would put it back.  With parallel rays there is no
source, hence no SOD/SDD, no magnification, no cone angle and no fan angle:
the projection at angle t is a property of the volume and t alone.  Feed this
a single-source CBCT model, one msCBCT source, or the full 8-source msCBCT
model and the outputs land on the SAME detector, pixel for pixel subtractable.

Every output row is an independent 2D parallel sinogram (v IS +z), so a
downstream reconstruction is plain per-slice FBP with no geometry file.

TWO SEPARATE THINGS: THE DETECTOR, AND THE FOV
-----------------------------------------------
1. THE DETECTOR is the full-panel equivalent of the real one, in mm.
   The scanner runs a HALF-DETECTOR (offset) geometry: at 0.4 mm the panel is
   372 x 287 px = 148.8 x 114.8 mm, shifted by ``offset_u`` = 70.5 mm, so it
   spans u in [-3.9, +144.9] mm.  Mirroring that about the axis gives the
   detector this file projects onto:

       FULL detector = 2 * ceil(u_far / pitch) cols  x  the panel's own rows
                     = 726 x 287 px = 290.4 x 114.8 mm at 0.4 mm
                     (1450 x 574 px = 290.0 x 114.8 mm at 0.2 mm)

   Width doubles because the offset is undone; HEIGHT IS UNCHANGED, because
   the half-detector trick is lateral only.  The output image therefore has
   the physical size a real full detector would have, which is what a
   downstream ASTRA reconstruction expects — it is NOT cropped to the FOV.

2. THE FOV is a property of the rays, not of the image.  Rebinning the fan to
   parallel, a ray at fan angle gamma is the parallel ray at signed distance
   ``s = SOD sin(gamma)`` from the rotation axis, so the panel's outer edge
   ray is tangent to the cylinder

       R_FOV = SOD * sin(atan(u_far / SDD)) = 93.31 mm

   That cylinder is what a 360 deg orbit actually measures (the panel reaches
   3.9 mm across the central ray, so every azimuth sees s from -2.58 to
   +93.31 mm, and a parallel ray (s, phi) is the same line as (-s, phi+180) —
   the negative half comes back from the opposite view).  ``fov_margin_xy``
   scales it.

   The volume is zeroed outside that cylinder before projecting, because an
   INR is unconstrained out there and cheerfully invents material a ray would
   otherwise integrate.  So the detector is deliberately WIDER (+/-145.2 mm)
   than the FOV (+/-93.3 mm): the outer columns come out zero, just as a real
   full panel would see nothing there.

NOTHING IS CONFIGURED THAT CAN BE DERIVED
------------------------------------------
The only free fields are ``num_angle`` (360 deg split evenly) and the two
FOV margins.  Everything else follows: the volume grid and ``rot_dir`` from
the training config, the detector from the panel tables in ``params.py``.

TWO FOV MARGINS
---------------
``fov_margin_xy`` scales the FOV cylinder's RADIUS, ``fov_margin_z`` its
HEIGHT, independently.  The two half-extents come from unrelated facts and so
want unrelated slack: the radius is a rebinning identity (the panel's outer
ray is tangent to ``R = SOD sin(atan(u_far / SDD))``, a hard edge, and past it
a 360 deg orbit measures nothing at all), while the z range is the cone
converging over the measured detector rows — a soft edge, where a little
extra keeps the last measured rows from being clipped, and too much drags in
the INR's invented material above and below the phantom.

THE ONE NUMBER THAT DESCRIBES THE CHECKPOINT
---------------------------------------------
``NORM_MM``.  The trained network eats NORMALIZED coordinates: training feeds
it ``xyz_mm / SOD`` (``R = SOD / pixel_size`` voxels — see ``train.py``).
That says how to READ THE CHECKPOINT, not how the data were taken.  The
scanner constants used above (panel size, offset_u, SOD, SDD) decide the
detector's physical size and where the volume is trustworthy; not one of them
touches an output ray direction.

CONVENTIONS (all verified against astra 2.5.0)
-----------------------------------------------
* World frame = the training pipeline's ASTRA volume frame: origin on the
  rotation axis at the volume centre, +z along the axis, mm.
* Volume array is ``[iz, iy, ix]`` and voxel ``(ix,iy,iz)`` sits at
  ``(i + 0.5 - n/2) * voxel_mm`` per axis — identical to
  ``geometry.astra_grid_chunks``, so this matches the training export.
* Angle ``t`` (times ``rot_dir``) gives ``u_hat = (cos t, sin t, 0)`` and ray
  direction ``(-sin t, cos t, 0)`` — the same ``u`` column and the same
  source->detector direction as the cone_vec vectors of ``geometry.py``, so
  parallel angle t and cone frame t view the volume from the same side.
* A point p lands at ``u_mm = p . u_hat``, ``v_mm = p.z``; detector index
  ``c = u_mm/pitch + (cols-1)/2``, ``r = v_mm/pitch + (rows-1)/2``.  Row r
  increases with +z, like the measured projections.
* ASTRA integrates in VOXEL units; mu is in cm^-1, so the line integral is
  ``fp * voxel_mm / 10``.  Same convention as training
  (``proj = voxel_size_cm * sum(mu)``), directly comparable to a measurement.
* Output ``.nii``: numpy ``(angle, u, v)`` -> on-disk dims ``(v, u, angle)``,
  the layout every projection file in this repository uses.  A
  ``*_geom.json`` sidecar carries the full output geometry, because a sinogram
  decoupled from its scanner has to describe itself.

Run:  python reproject_parallel.py [--config config.json] [--model M.pkl] [--out DIR]
"""

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import astra
import SimpleITK as sitk
from tqdm import tqdm

import geometry as geo
from params import SystemParams, BinnedGeometry


# =============================================================================
# 0. The one number that describes the checkpoint
# =============================================================================
NORM_MM = SystemParams().SOD          # 410.0 mm  <->  normalized radius 1.0


# =============================================================================
# 1. The FOV cylinder — what the rays reach at the rotation axis
# =============================================================================

def _axial_coverage_mm(geom: BinnedGeometry, config, verbose=True):
    """z the 8-source msCBCT acquisition measures at the rotation axis.

    The union over the sources of each band's cone-converged z range.
    ``data_j`` is a RAW-frame row range and the preprocessing flips the band
    after zeroing, so the MEASURED stored rows are the mirrored range — the
    same mirror ``geometry.rays_zero_deg`` applies; getting it wrong here
    would stretch the mask over sources 1/8's padding.

      per band  ~16.9 mm      union of all 8  ->  [-51.6, +52.4] mm
    """
    P = geom.params
    lo, hi = [], []
    for s in range(P.source_number):
        j0, j1 = geom.data_j[s]
        m0, m1 = geom.H - j1, geom.H - j0                  # stored, measured
        z_det = geom.z_det[s] + (np.array([m0, m1 - 1])
                                 - (geom.H - 1) / 2.0) * geom.pixel_size
        z_axis = geom.z_src[s] + (z_det - geom.z_src[s]) * (P.SOD / P.SDD)
        lo.append(z_axis[0])
        hi.append(z_axis[1])
    if verbose:
        print(f"[FOV] {P.source_number} msCBCT bands, each ~"
              f"{np.mean(np.array(hi) - np.array(lo)):.1f} mm tall at the axis; "
              f"union used")
    return float(min(lo)), float(max(hi))


def fov_cylinder(geom: BinnedGeometry, config, margin_xy=1.0, margin_z=1.0,
                 verbose=True):
    """The CYLINDER this acquisition measures at the ROTATION AXIS.

    Two independent half-extents, both shrunk from the panel by the beam:

    * RADIAL — a ray at fan angle gamma is the parallel ray at signed distance
      ``s = SOD sin(gamma)`` from the axis, so the panel's outer edge is
      tangent to ``R = SOD sin(atan(u_far / SDD))``.  Raises if the panel does
      not reach across the central ray, because the offset-detector
      completeness argument then fails and the measured region is an annulus.

    * AXIAL — the detector's z extent projected back to y = 0 through the
      source.  THE CONE CONVERGES, so this is always SMALLER than the panel:
      114.8 mm of panel becomes only 75.7 mm at the axis for a full-panel
      single-source CBCT.  Supplied by ``_axial_coverage_mm``, which is the
      one acquisition-specific piece of this file.

    ``margin_xy`` scales the radius and ``margin_z`` the height, separately:
    the two edges are unrelated — see the module docstring.  The z range is
    expanded about its OWN CENTRE so a margin > 1 still widens an asymmetric
    range — one msCBCT band sits far off-centre, and scaling its endpoints
    would move the whole band instead of growing it.

    Returns (radius_mm, z_lo_mm, z_hi_mm).
    """
    P = geom.params
    half_w = geom.cols * geom.pixel_size / 2.0
    u_near, u_far = geom.offset_u - half_w, geom.offset_u + half_w
    if u_near > 0.0:
        raise ValueError(
            f"the panel spans u = [{u_near:+.2f}, {u_far:+.2f}] mm and never "
            f"crosses the central ray: this offset-detector scan leaves an "
            f"unmeasured hole of radius "
            f"{P.SOD * np.sin(np.arctan(u_near / P.SDD)):.2f} mm around the axis")
    R = P.SOD * np.sin(np.arctan(u_far / P.SDD)) * margin_xy

    z_lo, z_hi = _axial_coverage_mm(geom, config, verbose=verbose)
    zc, zh = (z_hi + z_lo) / 2.0, (z_hi - z_lo) / 2.0
    z_lo, z_hi = zc - zh * margin_z, zc + zh * margin_z
    if verbose:
        print(f"[FOV] radial: outer ray at fan "
              f"{np.degrees(np.arctan(u_far / P.SDD)):.2f} deg -> R = "
              f"{R:.2f} mm (diameter {2 * R:.1f} mm) — margin_xy "
              f"{margin_xy:g}")
        print(f"[FOV] axial : z at the rotation axis [{z_lo:+.2f}, {z_hi:+.2f}] mm "
              f"(height {z_hi - z_lo:.1f} mm) — margin_z {margin_z:g}")
    return float(R), float(z_lo), float(z_hi)


# =============================================================================
# 2. The volume grid — read from the training config
# =============================================================================

@dataclass
class VolumeGrid:
    """The voxel grid mu is evaluated on.  Centred on the rotation axis."""

    size_mm: tuple            # (x, y, z) full extent, from config img_size_mm
    voxel_mm: float           # from config voxel_size (cm) * 10

    @classmethod
    def from_config(cls, config):
        f = config["file"]
        return cls(size_mm=tuple(f["img_size_mm"]),
                   voxel_mm=float(f["voxel_size"]) * 10.0)

    @property
    def shape(self):
        """(nx, ny, nz) — same rounding as the training pipeline."""
        return tuple(int(round(s / self.voxel_mm)) for s in self.size_mm)

    def axes_mm(self):
        """(x, y, z) coordinates of the voxel centres, mm."""
        return tuple((np.arange(n) + 0.5 - n / 2.0) * self.voxel_mm
                     for n in self.shape)

    def summary(self):
        nx, ny, nz = self.shape
        print(f"[grid] {nx} x {ny} x {nz} voxels @ {self.voxel_mm:.3f} mm "
              f"= {self.size_mm[0]:.1f} x {self.size_mm[1]:.1f} x "
              f"{self.size_mm[2]:.1f} mm  ({nx * ny * nz / 1e6:.1f} M voxels, "
              f"{nx * ny * nz * 4 / 2**30:.2f} GiB float32)")


# =============================================================================
# 3. The output geometry — the full-panel-equivalent detector
# =============================================================================

@dataclass
class ParallelGeometry:
    """A parallel-beam circular scan over a full turn onto a FULL detector.

    No source, no SOD, no magnification, no detector offset — a full detector
    is centred on the axis by definition.  Built by :func:`full_detector`.
    """

    num_angle: int
    rot_dir: int              # +1 matches the cone_vec pipeline
    det_row_count: int        # v, +z
    det_col_count: int        # u
    det_pitch_mm: float       # square pixels, = the panel pitch

    def angles_deg(self):
        """360 deg split evenly; endpoint excluded (0 and 360 are one view)."""
        return np.arange(self.num_angle, dtype=np.float64) * (360.0 / self.num_angle)

    def vectors(self, voxel_mm):
        """The (num_angle, 12) ``parallel3d_vec`` array, in VOXEL units.

        Row = [rayX rayY rayZ | dX dY dZ | uX uY uZ | vX vY vZ].
        ASTRA needs every length in units of the volume's voxel, which
        ``astra.create_vol_geom`` fixes at 1 — so mm are divided by voxel_mm
        here and nowhere else.
        """
        t = np.radians(self.angles_deg()) * self.rot_dir
        c, s = np.cos(t), np.sin(t)
        pitch = self.det_pitch_mm / voxel_mm

        V = np.zeros((t.size, 12))
        V[:, 0], V[:, 1] = -s, c                 # ray: the cone_vec src->det dir
        # V[:, 3:6] stays 0 — the detector is centred on the axis (FULL detector)
        V[:, 6], V[:, 7] = c * pitch, s * pitch  # u: in-plane, +x at t = 0
        V[:, 11] = pitch                         # v: always +z (parallel!)
        return V

    def half_width_mm(self):
        return self.det_col_count * self.det_pitch_mm / 2.0

    def half_height_mm(self):
        return self.det_row_count * self.det_pitch_mm / 2.0

    def as_dict(self):
        return {"num_angle": self.num_angle, "angle_range_deg": 360.0,
                "rot_dir": self.rot_dir, "det_row_count": self.det_row_count,
                "det_col_count": self.det_col_count,
                "det_pitch_mm": self.det_pitch_mm, "det_offset_mm": 0.0,
                "det_size_mm": [self.det_col_count * self.det_pitch_mm,
                                self.det_row_count * self.det_pitch_mm]}

    def summary(self):
        print(f"[detector] {self.num_angle} views, 360 deg / {self.num_angle} "
              f"= {360.0 / self.num_angle:.4f} deg apart (rot_dir {self.rot_dir:+d})")
        print(f"[detector] FULL: {self.det_col_count} cols (u) x "
              f"{self.det_row_count} rows (v, +z) @ {self.det_pitch_mm:.2f} mm "
              f"= {2 * self.half_width_mm():.1f} x {2 * self.half_height_mm():.1f} mm, "
              f"centred (u +/-{self.half_width_mm():.1f}, "
              f"v +/-{self.half_height_mm():.1f} mm)")


def full_detector(geom: BinnedGeometry, num_angle, rot_dir=+1, verbose=True):
    """The full-panel equivalent of the real half detector.

    u: the half detector spans ``[offset_u - W/2, offset_u + W/2]``; mirroring
       that about the axis needs half-width ``u_far``, i.e.
       ``2 * ceil(u_far / pitch)`` columns — the offset undone, ~2x wider.
    v: the panel's own row count, unchanged — the offset trick is lateral only.
    """
    W = geom.cols * geom.pixel_size
    u_far = geom.offset_u + W / 2.0
    cols = 2 * int(np.ceil(u_far / geom.pixel_size))
    pg = ParallelGeometry(num_angle=int(num_angle), rot_dir=int(rot_dir),
                          det_row_count=int(geom.rows),
                          det_col_count=int(cols),
                          det_pitch_mm=float(geom.pixel_size))
    if verbose:
        print(f"[detector] HALF (real): {geom.cols} x {geom.rows} px = "
              f"{W:.1f} x {geom.rows * geom.pixel_size:.1f} mm, offset_u "
              f"{geom.offset_u:.1f} mm -> u in "
              f"[{geom.offset_u - W / 2:+.1f}, {u_far:+.1f}] mm")
    return pg


# =============================================================================
# 4. Stage 1 — mu from a trained NAF/INR checkpoint
# =============================================================================

def volume_from_inr(model_path, config, grid: VolumeGrid, gpu=0,
                    chunk_z=16, infer_chunk=10_300_000):
    """Evaluate a trained network at every voxel centre of ``grid``.

    Returns (nz, ny, nx) float32 of mu [cm^-1] — the ASTRA volume layout,
    identical to what ``train.train`` writes to .nii.
    """
    import torch
    import tinycudann as tcnn

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    network = tcnn.NetworkWithInputEncoding(
        n_input_dims=3, n_output_dims=1,
        encoding_config=config["encoding"],
        network_config=config["network"]).to(device)
    network.load_state_dict(torch.load(model_path, map_location=device))
    network.eval()
    print(f"[INR] loaded {model_path}")

    nx, ny, nz = grid.shape
    R_norm = NORM_MM / grid.voxel_mm      # voxels per normalized unit
    print(f"[INR] querying at xyz_mm / {NORM_MM:.1f} mm  (R = {R_norm:.1f} voxels)")

    vol = np.zeros((nz, ny, nx), dtype=np.float32)
    with torch.no_grad():
        for xyz, z0, z1 in tqdm(
                geo.astra_grid_chunks(nx, ny, nz, R_norm=R_norm, chunk_z=chunk_z),
                total=int(np.ceil(nz / chunk_z)), desc="INR -> volume"):
            pts = torch.from_numpy(xyz).to(device)
            out = []
            for i0 in range(0, pts.shape[0], infer_chunk):
                out.append(network(pts[i0:i0 + infer_chunk])[:, 0].cpu())
            vol[z0:z1] = torch.cat(out).numpy().reshape(z1 - z0, ny, nx)
    del network
    torch.cuda.empty_cache()

    # A diverged training run saves a checkpoint full of NaN and this would
    # otherwise write a gigabyte of NaN to disk, which only shows up much later
    # as a reconstruction that will not converge.  Fail here instead.
    finite = np.isfinite(vol)
    if not finite.all():
        raise ValueError(
            f"{model_path} evaluates to NaN/Inf at "
            f"{(~finite).mean() * 100:.1f}% of the grid — that checkpoint is "
            f"broken (its training diverged), not the re-projection")
    print(f"[INR] mu range [{vol.min():.4f}, {vol.max():.4f}] cm^-1")
    return vol


# =============================================================================
# 5. Mask the volume to the FOV cylinder before projecting
# =============================================================================

def apply_fov_mask(vol, grid: VolumeGrid, radius_mm, z_lo_mm, z_hi_mm):
    """Zero everything outside the measured cylinder — radially AND axially.

    Not cosmetics: an INR is unconstrained outside the FOV and invents
    material there, which a ray would otherwise integrate into the output.
    Skipping the axial half puts a cloud above and below the phantom: a
    full-panel CBCT beam reaches only ~76 mm of z at the axis, but the
    detector spans 114.8 mm.
    """
    x, y, z = grid.axes_mm()
    keep_r = (x[None, :] ** 2 + y[:, None] ** 2) <= radius_mm ** 2   # (ny, nx)
    keep_z = (z >= z_lo_mm) & (z <= z_hi_mm)                         # (nz,)
    vol[:, ~keep_r] = 0.0
    vol[~keep_z] = 0.0
    print(f"[FOV] masked volume to r <= {radius_mm:.2f} mm and z in "
          f"[{z_lo_mm:+.2f}, {z_hi_mm:+.2f}] mm -> "
          f"{keep_r.mean() * keep_z.mean() * 100:.1f}% of the grid kept "
          f"({int(keep_z.sum())}/{len(z)} slices; the grid spans "
          f"r <= {min(grid.size_mm[0], grid.size_mm[1]) / 2:.0f} mm, "
          f"|z| <= {grid.size_mm[2] / 2:.0f} mm)")
    return vol


# =============================================================================
# 6. Stage 2 — ASTRA parallel forward projection
# =============================================================================

def forward_project(vol, grid: VolumeGrid, pg: ParallelGeometry,
                    angle_chunk=None, to_cm=True, gpu=None):
    """Parallel-beam line integrals of ``vol``.

    Returns (num_angle, det_row_count, det_col_count) float32.  With
    ``to_cm`` (the default) the values are dimensionless line integrals
    ``int mu dl`` with mu in cm^-1 and dl in cm — the quantity the training
    data holds.  With ``to_cm=False`` they are in voxel units, which is what
    ASTRA natively returns and what a reconstruction wants back.
    """
    nx, ny, nz = grid.shape
    if vol.shape != (nz, ny, nx):
        raise ValueError(f"volume is {vol.shape}, grid expects {(nz, ny, nx)}")

    vol_geom = astra.create_vol_geom(ny, nx, nz)     # (rows y, cols x, slices z)
    V = pg.vectors(grid.voxel_mm)
    rows, cols = pg.det_row_count, pg.det_col_count
    n = pg.num_angle
    chunk = int(angle_chunk or n)

    sino = np.zeros((n, rows, cols), dtype=np.float32)
    print(f"[FP] {n} parallel views of {nx}x{ny}x{nz} -> "
          f"{n} x {rows} x {cols} ({sino.nbytes / 2**30:.2f} GiB), "
          f"{int(np.ceil(n / chunk))} chunk(s)")

    vol_id = astra.data3d.create('-vol', vol_geom, vol)
    try:
        for a0 in tqdm(range(0, n, chunk), desc="ASTRA parallel FP"):
            a1 = min(a0 + chunk, n)
            proj_geom = astra.create_proj_geom('parallel3d_vec', rows, cols,
                                               V[a0:a1])
            sino_id = astra.data3d.create('-proj3d', proj_geom, 0)
            cfg = astra.astra_dict('FP3D_CUDA')
            cfg['VolumeDataId'] = vol_id
            cfg['ProjectionDataId'] = sino_id
            if gpu is not None:
                cfg['option'] = {'GPUindex': int(gpu)}
            alg_id = astra.algorithm.create(cfg)
            try:
                astra.algorithm.run(alg_id)
                # ASTRA 3D projection layout is (v, angle, u)
                sino[a0:a1] = astra.data3d.get(sino_id).transpose(1, 0, 2)
            finally:
                astra.algorithm.delete(alg_id)
                astra.data3d.delete(sino_id)
    finally:
        astra.data3d.delete(vol_id)

    if to_cm:
        # ASTRA integrates in voxel units; mu is cm^-1  ->  * voxel_mm / 10.
        sino *= grid.voxel_mm / 10.0
    print(f"[FP] line integral range [{sino.min():.4f}, {sino.max():.4f}]"
          + (" (dimensionless)" if to_cm else " (voxel units)"))
    return sino


# =============================================================================
# 7. Optional: reconstruct the re-projection back, on the same geometry
# =============================================================================

def sirt_reconstruct(sino_cm, grid: VolumeGrid, pg: ParallelGeometry,
                     iterations=100, gpu=None):
    """SIRT of the parallel sinogram on the SAME geometry object, in cm^-1.

    Self-consistency check, and a worked example of the geometry a downstream
    reconstruction needs: ``parallel3d_vec`` with ``pg.vectors(voxel_mm)`` and
    the sinogram divided back to voxel units.
    """
    nx, ny, nz = grid.shape
    vol_geom = astra.create_vol_geom(ny, nx, nz)
    proj_geom = astra.create_proj_geom('parallel3d_vec', pg.det_row_count,
                                       pg.det_col_count, pg.vectors(grid.voxel_mm))
    data = np.ascontiguousarray(
        (sino_cm / (grid.voxel_mm / 10.0)).transpose(1, 0, 2))   # (v, angle, u)
    rec_id = astra.data3d.create('-vol', vol_geom, 0)
    sino_id = astra.data3d.create('-proj3d', proj_geom, data)
    cfg = astra.astra_dict('SIRT3D_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sino_id
    cfg['option'] = {'MinConstraint': 0.0}
    if gpu is not None:
        cfg['option']['GPUindex'] = int(gpu)
    alg_id = astra.algorithm.create(cfg)
    try:
        print(f"[SIRT] {iterations} iterations on the re-projected sinogram ...")
        astra.algorithm.run(alg_id, int(iterations))
        rec = astra.data3d.get(rec_id)
    finally:
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(sino_id)
    return rec.astype(np.float32)          # (nz, ny, nx), cm^-1


# =============================================================================
# 8. Output
# =============================================================================

def save_projection(sino, grid, pg, out_path, out_name, extra=None):
    """Write ``<out_name>.nii`` + ``<out_name>_geom.json``.

    numpy (angle, v, u) -> transposed to (angle, u, v) so the on-disk NIfTI
    dims are (v, u, angle): the layout every projection file in this
    repository uses, i.e. sitk reads it back as (num_proj, cols, rows)
    exactly like ``dataset.py`` reads the measured data.
    """
    os.makedirs(out_path, exist_ok=True)
    p = pg.det_pitch_mm
    img = sitk.GetImageFromArray(np.ascontiguousarray(sino.transpose(0, 2, 1)))
    img.SetSpacing([p, p, 1.0])            # dims (v, u, angle)
    nii = os.path.join(out_path, f"{out_name}.nii")
    sitk.WriteImage(img, nii)

    meta = {"beam": "parallel3d",
            "generator": "reproject_parallel.py",
            "astra_version": astra.__version__,
            "detector": "full (centred, no offset)",
            "geometry": pg.as_dict(),
            "angles_deg": pg.angles_deg().tolist(),
            "volume_grid": {"size_mm": list(grid.size_mm),
                            "voxel_mm": grid.voxel_mm,
                            "shape_xyz": list(grid.shape)},
            "units": "dimensionless line integral (mu[cm^-1] * length[cm])",
            "array_layout": {"nii_dims": "(v, u, angle)",
                             "sitk_numpy": "(angle, u, v)"}}
    if extra:
        meta.update(extra)
    with open(os.path.join(out_path, f"{out_name}_geom.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[save] {nii}  (dims v={pg.det_row_count}, u={pg.det_col_count}, "
          f"angle={pg.num_angle})")
    print(f"[save] {os.path.join(out_path, out_name + '_geom.json')}")
    return nii


def preview_png(sino, pg, out_path, out_name):
    """Four views + the central-row sinogram — a 5 s sanity look."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ang = pg.angles_deg()
    idx = [int(round(f * pg.num_angle)) % pg.num_angle for f in (0, .25, .5, .75)]
    vmax = float(np.percentile(sino[::max(1, pg.num_angle // 16)], 99.8))
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.6))
    for ax, i in zip(axes[:4], idx):
        ax.imshow(sino[i], cmap="gray", vmin=0, vmax=vmax, origin="lower",
                  aspect="auto")
        ax.set_title(f"{ang[i]:.1f} deg", fontsize=9)
        ax.set_xlabel("u"); ax.set_ylabel("v (+z up)")
    r = pg.det_row_count // 2
    im = axes[4].imshow(sino[:, r, :], cmap="gray", vmin=0, vmax=vmax,
                        aspect="auto")
    axes[4].set_title(f"sinogram, v row {r}", fontsize=9)
    axes[4].set_xlabel("u"); axes[4].set_ylabel("view")
    fig.colorbar(im, ax=axes[4], shrink=0.85)
    fig.tight_layout()
    png = os.path.join(out_path, f"{out_name}_preview.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"[save] {png}")


# =============================================================================
# 9. Entry point
# =============================================================================

def reproject_parallel(config, rc):
    """``config`` = the training config (grid, rot_dir, network shape, the
    default checkpoint and output folder); ``rc`` = the re-projection config
    (``num_angle``, the two FOV margins, output).
    """
    f = config["file"]
    model_path = rc.get("model_path") or os.path.join(
        f["model_dir"], f"model_{f['tag']}_{config['train']['epoch']}.pkl")
    if not os.path.isfile(model_path):
        d = os.path.dirname(model_path) or "."
        have = sorted(n for n in os.listdir(d) if n.endswith(".pkl")) \
            if os.path.isdir(d) else []
        raise FileNotFoundError(
            f"no checkpoint at {model_path}"
            + ("; that folder holds: " + ", ".join(have) if have else
               f"; {d} holds no .pkl"))

    grid = VolumeGrid.from_config(config)
    grid.summary()

    # the scanner's own calibration: sets the detector's physical size and the
    # FOV radius — never an output ray direction
    P = SystemParams()
    geom = BinnedGeometry(P, grid.voxel_mm / P.pixel_size)
    gpu = rc.get("gpu") if rc.get("gpu") is not None \
        else config["train"].get("gpu", 0)

    # ---- stage 1: mu on the grid, masked to the measured cylinder ----------
    vol = volume_from_inr(model_path, config, grid, gpu=gpu,
                          chunk_z=config["train"].get("grid_chunk_size", 16),
                          infer_chunk=config["train"].get("infer_chunk_size",
                                                          10_300_000))
    margin_xy, margin_z = rc["fov_margin_xy"], rc["fov_margin_z"]
    R_fov, z_lo, z_hi = fov_cylinder(geom, config,
                                     margin_xy=margin_xy, margin_z=margin_z)
    apply_fov_mask(vol, grid, R_fov, z_lo, z_hi)

    # ---- stage 2: the full-panel-equivalent detector -----------------------
    pg = full_detector(geom, rc["num_angle"], rot_dir=f.get("rot_dir", +1))
    pg.summary()
    if pg.half_height_mm() < max(abs(z_lo), abs(z_hi)):
        print(f"[detector] note: the panel is {2 * pg.half_height_mm():.1f} mm "
              f"tall (v +/-{pg.half_height_mm():.1f} mm) but the beam reaches "
              f"z in [{z_lo:+.1f}, {z_hi:+.1f}] mm — part of the FOV falls "
              f"outside the detector")

    sino = forward_project(vol, grid, pg, angle_chunk=rc.get("angle_chunk"),
                           gpu=gpu)

    # ---- output ------------------------------------------------------------
    out_path = rc.get("out_path") or f["out_dir"]
    out_name = rc.get("out_name") or (os.path.splitext(os.path.basename(model_path))[0]
                                     .removeprefix("model_") + f"_par3d_{pg.num_angle}")
    save_projection(sino, grid, pg, out_path, out_name,
                    extra={"source_model": model_path,
                           "inr_norm_mm": NORM_MM,
                           "fov_radius_mm": R_fov,
                           "fov_z_mm": [z_lo, z_hi],
                           "fov_margin_xy": margin_xy,
                           "fov_margin_z": margin_z})
    if rc.get("save_volume", False):
        v = sitk.GetImageFromArray(vol)
        v.SetSpacing([grid.voxel_mm] * 3)
        sitk.WriteImage(v, os.path.join(out_path, f"{out_name}_volume.nii"))
        print(f"[save] {out_path}/{out_name}_volume.nii")
    if rc.get("preview", True):
        preview_png(sino, pg, out_path, out_name)

    if rc.get("verify_sirt", 0):
        rec = sirt_reconstruct(sino, grid, pg, rc["verify_sirt"], gpu)
        m = vol > 0
        num = float(np.abs(rec[m] - vol[m]).mean())
        print(f"[verify] SIRT of the re-projection vs the projected volume: "
              f"mean |diff| = {num:.5f} cm^-1 "
              f"({num / float(np.abs(vol[m]).mean()) * 100:.2f}% of the level)")
        r = sitk.GetImageFromArray(rec)
        r.SetSpacing([grid.voxel_mm] * 3)
        sitk.WriteImage(r, os.path.join(out_path, f"{out_name}_sirt.nii"))
        print(f"[save] {out_path}/{out_name}_sirt.nii")

    return sino, grid, pg


# =============================================================================
# 10. Defaults for a run
# =============================================================================

REPROJECT_CONFIG = {
    "model_path":    None,     # None -> <model_dir>/model_<tag>_<train.epoch>.pkl (final checkpoint)
    # ---- the only geometric choices ----------------------------------------
    "num_angle":     1440,     # 360 deg split evenly into this many views
    "fov_margin_xy": 1.2,      # scales the FOV cylinder's RADIUS (mask only)
    "fov_margin_z":  1.05,     # scales its HEIGHT, independently
    # ---- run / output --------------------------------------------------------
    "out_path":      None,     # None -> the training config's out_dir
    "out_name":      None,     # None -> <model file stem without "model_">_par3d_<num_angle>
    "angle_chunk":   120,      # views per ASTRA call (GPU memory)
    "save_volume":   False,    # also write the masked mu volume (<out_name>_volume.nii)
    "preview":       True,     # <out_name>_preview.png
    "verify_sirt":   0,        # >0: SIRT the result back, report the error
    "gpu":           None,     # None -> the training config's train.gpu
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="config.json",
                    help="training config (grid, rot_dir, network shape)")
    ap.add_argument("--model", default=None, help="override model_path")
    ap.add_argument("--out", default=None, help="override out_path")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    rc = dict(REPROJECT_CONFIG)
    if args.model:
        rc["model_path"] = args.model
    if args.out:
        rc["out_path"] = args.out
    reproject_parallel(cfg, rc)
