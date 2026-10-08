"""astra_recon.py — the ASTRA reconstructions behind the notebooks in this folder.

* ``ConeBeamReconstruction`` — SIRT of a MEASURED scan (conventional CBCT or
  msCBCT) on the exact ASTRA cone_vec geometry of the scanner, i.e. the same
  geometry ../CBCT and ../msCBCT train on, so the volumes overlay voxel for
  voxel.
* ``ParallelReconstruction`` — FBP / SIRT of a parallel-beam RE-PROJECTION
  written by ../CBCT or ../msCBCT reproject_parallel.py.

No sidecar file is used as input.  Everything is rebuilt from ``params.py``
(the scanner), the .nii itself (frame count, frame size, pixel width from the
header) and the few per-scan numbers of the notebook's parameter cell
(angular span / start / rotation direction).

ARRAY LAYOUTS
-------------
    on disk .nii      : (v, u, frame)     <- preprocessing / re-projection output
    SimpleITK numpy   : (frame, u, v)
    self.proj         : (frame, v, u)     row v increases with +z
    ASTRA projections : (v, frame, u)
    volume            : (z, y, x)  ->  saved as .nii dims (x, y, z), voxel width in mm
    mu                : cm^-1, i.e. ``rec * 10 / voxel_mm`` (ASTRA works in voxels)
"""

import json
import time
from pathlib import Path

import numpy as np
import astra
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm

from params import SystemParams, BinnedGeometry

FBP_FILTERS = ("ram-lak", "shepp-logan", "cosine", "hamming", "hann")


# =============================================================================
# Shared helpers
# =============================================================================

def load_projections(path):
    """Read a projection .nii -> ((frame, v, u) float32, detector pitch [mm])."""
    img = sitk.ReadImage(str(path))
    if img.GetDimension() != 3:
        raise ValueError(f"expected a 3D projection stack, got a "
                         f"{img.GetDimension()}D image")
    pitch_v, pitch_u = img.GetSpacing()[:2]
    if abs(pitch_v - pitch_u) > 1e-6:
        raise ValueError(f"detector pixels are not square: header spacing "
                         f"(v, u) = ({pitch_v}, {pitch_u}) mm")
    proj = np.ascontiguousarray(
        sitk.GetArrayFromImage(img).transpose(0, 2, 1), np.float32)
    return proj, float(pitch_u)


def save_volume(rec, path, voxel_mm):
    """Write a (z, y, x) volume as float32 .nii, dims (x, y, z), spacing in mm."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(rec.astype(np.float32))
    img.SetSpacing([float(voxel_mm)] * 3)
    sitk.WriteImage(img, str(path))
    print(f"saved {path}  (x, y, z) = {rec.shape[::-1]}")
    return path


def plot_slices(rec, title, pct=99.8):
    """Central axial / coronal / sagittal slices of a (z, y, x) volume, +z up."""
    nz, ny, nx = rec.shape
    vmax = float(np.percentile(rec[::2, ::2, ::2], pct))
    views = [(rec[nz // 2], "axial (z mid)", "upper"),
             (rec[:, ny // 2, :], "coronal (y mid)", "lower"),
             (rec[:, :, nx // 2], "sagittal (x mid)", "lower")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (img, name, origin) in zip(axes, views):
        im = ax.imshow(img, cmap="gray", vmin=0.0, vmax=vmax, origin=origin)
        ax.set_title(f"{name}  [{title}]", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8, label="mu [cm^-1]")
    fig.tight_layout()
    plt.show()


def _astra_run(alg, proj_geom, vol_geom, proj_astra, iterations=None, gpu=None):
    """Run one ASTRA 3D algorithm; ``proj_astra`` is (v, frame, u).

    Returns the (z, y, x) volume in voxel units.
    """
    rec_id = astra.data3d.create('-vol', vol_geom, 0)
    proj_id = astra.data3d.create('-proj3d', proj_geom,
                                  np.ascontiguousarray(proj_astra))
    cfg = astra.astra_dict(alg)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    cfg['option'] = {}
    if alg == 'SIRT3D_CUDA':
        cfg['option']['MinConstraint'] = 0.0        # non-negativity
    if gpu is not None:
        cfg['option']['GPUindex'] = int(gpu)
    alg_id = astra.algorithm.create(cfg)
    try:
        if iterations:
            astra.algorithm.run(alg_id, int(iterations))
        else:
            astra.algorithm.run(alg_id)
        return astra.data3d.get(rec_id)             # (z, y, x)
    finally:
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(proj_id)


def _stem(path):
    """File name before ".nii", without a leading "proj_"."""
    return Path(path).name.split(".nii")[0].removeprefix("proj_")


class _Reconstruction:
    """Volume grid, display and output shared by both reconstructions."""

    @property
    def volume_shape(self):
        """(nx, ny, nz) — same rounding as the NAF training grid."""
        return tuple(np.round(self.img_size / self.voxel_size).astype(int))

    def show_slices(self):
        if self.rec_cm is None:
            raise RuntimeError("run() first")
        plot_slices(self.rec_cm, self.label)

    def save(self, out_dir):
        """Write ``<out_dir>/recon_<name>_<label>.nii`` (mu in cm^-1)."""
        if self.rec_cm is None:
            raise RuntimeError("run() first")
        return save_volume(self.rec_cm,
                           Path(out_dir) / f"recon_{self.name}_{self.label}.nii",
                           self.voxel_size)


# =============================================================================
# Cone beam: SIRT of a measured scan
# =============================================================================

class ConeBeamReconstruction(_Reconstruction):
    """SIRT on the ASTRA ``cone_vec`` geometry of the real scanner.

    "CBCT"   — one source on the rotation axis, the full flat panel; frame i
               sits at ``start_deg + i * span_deg / N``.
    "msCBCT" — 8 sources along z firing in turn, each onto its own 65-row
               band; frame k is fired by source ``k % 8`` at
               ``start_deg + (k // 8) * span_deg / (N / 8) + (k % 8) * deg_per_frame``
               (the gantry keeps rotating between the sources of a view).
    """

    def __init__(self, proj, pitch_mm, acquisition, span_deg=360.0,
                 start_deg=0.0, rot_dir=+1, img_size=(240.0, 240.0, 120.0),
                 voxel_size=None, name="recon"):
        """``proj``: (frame, v, u) float32 line integrals, row v increasing
        with +z.  ``pitch_mm``: detector pixel width (header spacing)."""
        if acquisition not in ("CBCT", "msCBCT"):
            raise ValueError(f'acquisition must be "CBCT" or "msCBCT", '
                             f"got {acquisition!r}")
        P = SystemParams()
        b = pitch_mm / P.pixel_size
        if b < 1 or abs(b - round(b)) > 1e-3:
            raise ValueError(f"pixel width {pitch_mm} mm is not an integer "
                             f"multiple of the native {P.pixel_size} mm "
                             f"(bin = {b:.3f})")
        g = self.geom = BinnedGeometry(P, int(round(b)))
        self.proj = np.ascontiguousarray(proj, np.float32)
        n = self.proj.shape[0]
        frame = (g.rows if acquisition == "CBCT" else g.H, g.cols)
        if self.proj.shape[1:] != frame:
            raise ValueError(f"frames are {self.proj.shape[1:]} (v, u), a "
                             f"{acquisition} scan at {g.pixel_size:.2f} mm has "
                             f"{frame} — wrong file for this notebook?")

        # ---- per-frame source / detector z and gantry angle ----
        if acquisition == "CBCT":
            self.z_src = np.zeros(n)
            self.z_det = np.full(n, P.cbct_offset_v)
            self.angle_deg = start_deg + np.arange(n) * (span_deg / n)
        else:
            ns = P.source_number
            if n % ns:
                raise ValueError(f"{n} frames is not a multiple of the "
                                 f"{ns} sources")
            src = np.arange(n) % ns
            self.z_src, self.z_det = g.z_src[src], g.z_det[src]
            self.angle_deg = (start_deg + (np.arange(n) // ns) * (span_deg / (n // ns))
                              + src * P.deg_per_frame)

        self.acquisition = acquisition
        self.span_deg, self.start_deg = span_deg, start_deg
        self.rot_dir = int(rot_dir)
        self.voxel_size = float(voxel_size or g.pixel_size)
        self.img_size = np.asarray(img_size, float)
        self.name = name
        self.rec_cm = None
        self.label = "unrun"

    @classmethod
    def from_file(cls, path, acquisition, **kw):
        """Load one projection .nii; the bin factor comes from its header."""
        print(f"projection : {path}")
        proj, pitch = load_projections(path)
        rec = cls(proj, pitch, acquisition, name=_stem(path), **kw)
        rec.summary()
        return rec

    def summary(self):
        g, n = self.geom, self.proj.shape[0]
        ns = g.params.source_number
        nx, ny, nz = self.volume_shape
        print(f"geometry   : params.py + header pixel width {g.pixel_size:.2f} mm "
              f"-> bin {g.b}, {self.acquisition} frame "
              f"{self.proj.shape[1]} x {self.proj.shape[2]} (v x u)")
        print(f"frames     : {n} @ {g.pixel_size:.2f} mm  "
              f"({self.proj.nbytes / 1e9:.2f} GB)")
        if self.acquisition == "CBCT":
            print(f"angles     : {n} frames x {self.span_deg / n:.4f} deg "
                  f"(span {self.span_deg:g} deg, start {self.start_deg:g} deg, "
                  f"rot_dir {self.rot_dir:+d}) — must match the scan")
        else:
            print(f"angles     : {n // ns} views x {ns} sources, view step "
                  f"{self.span_deg / (n // ns):.4f} deg, source step "
                  f"{g.params.deg_per_frame:g} deg (span {self.span_deg:g} deg, "
                  f"start {self.start_deg:g} deg, rot_dir {self.rot_dir:+d}) "
                  f"— must match the scan")
        limit = 6.0 if self.acquisition == "CBCT" else 8.0
        print(f"line integr: max = {self.proj.max():.3f}, "
              f"mean(>0.01) = {self.proj[self.proj > 0.01].mean():.3f}"
              + ("   ! max is large: frames without beam may be left in the file"
                 if self.proj.max() > limit else ""))
        print(f"volume     : {nx} x {ny} x {nz} @ {self.voxel_size:.2f} mm "
              f"= {self.img_size[0]:.0f} x {self.img_size[1]:.0f} x "
              f"{self.img_size[2]:.0f} mm  ({nx * ny * nz * 4 / 1e9:.2f} GB per copy)")
        return self

    def vectors(self):
        """The (N, 12) ``cone_vec`` array, in VOXEL units.

        Row = [src | det centre | u step | v step].  Source at
        ``(sin t * SOD, -cos t * SOD, z_src)``, detector centre ODD beyond the
        axis, shifted by the half-fan ``offset_u`` along u, v along +z.  The
        same construction as ../CBCT/geometry.py base_vector and
        ../msCBCT/geometry.py cone_vec_vectors.
        """
        g, vx = self.geom, self.voxel_size
        P = g.params
        t = np.radians(self.angle_deg) * self.rot_dir
        sod, odd = P.SOD / vx, P.ODD / vx
        off_u, off_v = g.offset_u / vx, g.offset_v / vx
        pitch = g.pixel_size / vx                    # detector pitch in voxels

        V = np.zeros((t.size, 12))
        V[:, 0], V[:, 1], V[:, 2] = np.sin(t) * sod, -np.cos(t) * sod, self.z_src / vx
        V[:, 3] = -np.sin(t) * odd + np.cos(t) * off_u
        V[:, 4] = np.cos(t) * odd + np.sin(t) * off_u
        V[:, 5] = self.z_det / vx + off_v
        V[:, 6], V[:, 7] = np.cos(t) * pitch, np.sin(t) * pitch   # u: horizontal
        V[:, 11] = pitch                                          # v: along +z
        return V

    def run(self, iterations=300, gpu=None):
        """SIRT (non-negative) -> ``self.rec_cm``, (z, y, x) in cm^-1."""
        n, rows, cols = self.proj.shape
        nx, ny, nz = self.volume_shape
        proj_geom = astra.create_proj_geom("cone_vec", rows, cols, self.vectors())
        vol_geom = astra.create_vol_geom(ny, nx, nz)   # (rows y, cols x, slices z)
        print(f"reconstructing {n} frames with SIRT3D_CUDA, {iterations} "
              f"iterations -> volume {nx} x {ny} x {nz} @ {self.voxel_size:.2f} mm ...")
        t0 = time.time()
        rec = _astra_run("SIRT3D_CUDA", proj_geom, vol_geom,
                         self.proj.transpose(1, 0, 2), iterations, gpu)
        # voxel^-1 -> cm^-1
        self.rec_cm = rec * 10.0 / self.voxel_size
        self.label = f"SIRT_it{iterations}"
        print(f"   done in {time.time() - t0:.1f} s;  mu range "
              f"[{self.rec_cm.min():.3f}, {self.rec_cm.max():.3f}] cm^-1")
        return self


# =============================================================================
# Parallel beam: FBP / SIRT of a re-projection
# =============================================================================

def _ramp_response(n_pad, ds, window):
    """Frequency response of the discrete ramp filter, sample spacing ``ds``.

    Built from the SPATIAL Ram-Lak kernel
        h(0) = 1 / (4 ds^2),  h(k odd) = -1 / (pi k ds)^2,  h(k even) = 0,
    whose DFT times ``ds`` (the convolution's sample spacing) is the discrete
    equivalent of ``|w|`` with the correct DC term.  ``ds`` is expressed in the
    units the reconstruction grid uses, i.e. VOXELS.
    """
    if window not in FBP_FILTERS:
        raise ValueError(f"fbp_filter must be one of {FBP_FILTERS}, got {window!r}")
    h = np.zeros(n_pad)
    h[0] = 0.25 / ds ** 2
    odd = np.arange(1, n_pad // 2 + 1, 2)
    h[odd] = -1.0 / (np.pi * odd * ds) ** 2
    h[n_pad - odd] = h[odd]
    H = np.real(np.fft.rfft(h)) * ds

    f = np.fft.rfftfreq(n_pad, d=ds)
    fn = f / f[-1]                              # 0 .. 1 (1 = Nyquist)
    if window == "shepp-logan":
        H = H * np.sinc(fn / 2.0)
    elif window == "cosine":
        H = H * np.cos(np.pi * fn / 2.0)
    elif window == "hamming":
        H = H * (0.54 + 0.46 * np.cos(np.pi * fn))
    elif window == "hann":
        H = H * (0.5 + 0.5 * np.cos(np.pi * fn))
    return H


class ParallelReconstruction(_Reconstruction):
    """FBP / SIRT on the ASTRA ``parallel3d_vec`` geometry.

    WHY THIS IS SO MUCH SMALLER THAN THE CONE VERSION
    --------------------------------------------------
    The cone version has to rebuild the whole scanner — 8 sources, their z
    tables, the half-fan offset, the per-frame source phase — because every
    one of those changes a ray.  A parallel-beam projection has none of it: at
    angle ``t`` the rays are ``(-sin t, cos t, 0)`` and that is the entire
    geometry.  So the only things needed are

    * what the ``.nii`` already carries — view count, detector rows / cols and
      the pitch (header spacing), which is also the default voxel size;
    * SOD / SDD — used for ONE number, the measured FOV radius, and never for
      a ray direction (see :attr:`fov_radius_mm`);
    * the angular span / start / direction, from the parameter cell.

    ALGORITHMS
    ----------
    ``SIRT`` is ASTRA's own 3D algorithm on the ``parallel3d_vec`` geometry.
    ``FBP`` is ramp-filter-along-u followed by ``BP3D_CUDA``, which for
    parallel rays IS filtered backprojection (the rays are perpendicular to z,
    so each detector row is an independent 2D parallel sinogram of its own
    plane — no cone weighting, no redundancy weighting, no Parker window).
    Doing it in 3D rather than slice by slice keeps the output on the
    requested voxel grid instead of on the detector's row grid.

    The two normalisations were measured against astra 2.5.0, not guessed:

    * ``BP3D_CUDA`` on ``parallel3d_vec`` accumulates the interpolated
      detector value with weight ``1 / (pitch_u * pitch_v)`` per view
      (pitches in voxel units) — exact to 6 decimals for pitches 0.5 / 1 / 2
      in u and in v;
    * with that undone, ``FBP = (pitch_u * pitch_v) * dphi * BP(ramp(p))``,
      where ``dphi`` integrates over pi of unique angles (half of a 360 deg
      orbit, because a parallel ray ``(s, phi)`` is the same line as
      ``(-s, phi+180)``).

    The ramp itself is the DFT of the spatial Ram-Lak kernel, not a bare
    ``|w|``: the bare version leaves a ~0.5% negative bias in the
    reconstructed level.  The Ram-Lak version reproduces astra's own 2D
    ``FBP_CUDA`` to 5 decimals.
    """

    def __init__(self, proj, det_pitch, span_deg=360.0, start_deg=0.0,
                 rot_dir=+1, img_size=None, voxel_size=None, fov_margin=1.3,
                 name="par3d", source=None):
        """``proj``: (view, det_row v, det_col u) float32 line integrals,
        dimensionless (mu[cm^-1] * length[cm]); row v increases with +z and the
        detector is centred on the rotation axis (a FULL detector, no offset).
        ``det_pitch``: detector pixel width [mm], square pixels."""
        self.params = SystemParams()
        self.proj = np.ascontiguousarray(proj, np.float32)
        if self.proj.ndim != 3:
            raise ValueError(f"expected a 3D (view, v, u) stack, got {self.proj.shape}")
        self.det_pitch = float(det_pitch)
        self.source = Path(source) if source else None
        self.name = name
        self.span_deg, self.start_deg = span_deg, start_deg
        self.rot_dir = int(rot_dir)
        self.fov_margin = float(fov_margin)
        self.view_angle_deg = start_deg + np.arange(self.n_views, dtype=np.float64) \
            * (span_deg / self.n_views)
        self.voxel_size = float(voxel_size or self.det_pitch)
        self.img_size = np.asarray(img_size if img_size is not None
                                   else self._default_img_size(), float)
        self.rec_cm = None
        self.label = "unrun"

    # ------------------------------------------------------------------ #
    #  what the projection file already tells us
    # ------------------------------------------------------------------ #
    @property
    def n_views(self):
        return self.proj.shape[0]

    @property
    def det_rows(self):
        return self.proj.shape[1]

    @property
    def det_cols(self):
        return self.proj.shape[2]

    @property
    def u_far_mm(self):
        """Detector half width [mm] -- it is centred, so this is its outer u."""
        return self.det_cols * self.det_pitch / 2.0

    @property
    def det_half_height_mm(self):
        return self.det_rows * self.det_pitch / 2.0

    @property
    def fov_radius_mm(self):
        """Radius at the rotation axis the measured rays reach, times the margin.

        The ONLY place SOD / SDD are used.  The re-projection came from a fan
        beam: a ray at fan angle ``gamma`` is the parallel ray at signed
        distance ``s = SOD sin(gamma)`` from the axis, so the detector's outer
        column is tangent to a cylinder of radius ``SOD sin(atan(u_far/SDD))``.
        Outside it the sinogram is zero by construction, so nothing is lost by
        not reconstructing there -- and the algorithms stop smearing the
        unmeasured corners back into the volume.
        """
        P = self.params
        return float(P.SOD * np.sin(np.arctan(self.u_far_mm / P.SDD))
                     * self.fov_margin)

    def _default_img_size(self):
        """(FOV diameter, FOV diameter, detector height), snapped to the voxel."""
        vx = self.voxel_size
        nxy = 2 * int(np.ceil(self.fov_radius_mm / vx))
        nz = int(round(self.det_rows * self.det_pitch / vx))
        return (nxy * vx, nxy * vx, nz * vx)

    # ------------------------------------------------------------------ #
    #  construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_file(cls, path, **kw):
        """Load one parallel-beam projection .nii.

        Everything geometric comes from the file (views, rows, cols, pitch)
        except the angular span / start / direction.  A ``*_geom.json`` written
        by the re-projection is only ever CROSS-CHECKED against those, never
        used as an input.
        """
        print(f"projection : {path}")
        proj, pitch = load_projections(path)
        rec = cls(proj, pitch, name=_stem(path), source=path, **kw)
        rec.summary()
        rec.crosscheck_sidecar()
        return rec

    def summary(self):
        n, rows, cols = self.proj.shape
        nx, ny, nz = self.volume_shape
        step = self.span_deg / n
        P = self.params
        print(f"views      : {n} views, from {self.start_deg:g} deg every "
              f"{step:.4f} deg, span {self.span_deg:g} deg (rot_dir {self.rot_dir:+d})")
        print(f"detector   : {cols} (u) x {rows} (v, +z) px @ {self.det_pitch:.2f} mm "
              f"= {cols * self.det_pitch:.1f} x {rows * self.det_pitch:.1f} mm, "
              f"centred, no offset  ({self.proj.nbytes / 1e9:.2f} GB)")
        print(f"FOV        : u up to +/-{self.u_far_mm:.1f} mm, SOD/SDD = "
              f"{P.SOD:.0f}/{P.SDD:.0f} mm -> R = SOD*sin(atan(u/SDD)) x "
              f"{self.fov_margin:g} = {self.fov_radius_mm:.2f} mm "
              f"(diameter {2 * self.fov_radius_mm:.1f} mm)")
        print(f"line integr: max = {self.proj.max():.3f}, "
              f"mean(>0.01) = {self.proj[self.proj > 0.01].mean():.3f}")
        print(f"volume     : {nx} x {ny} x {nz} @ {self.voxel_size:.2f} mm "
              f"= {self.img_size[0]:.1f} x {self.img_size[1]:.1f} x "
              f"{self.img_size[2]:.1f} mm  ({nx * ny * nz * 4 / 1e9:.2f} GB per copy)")
        if nz * self.voxel_size > 2 * self.det_half_height_mm + 1e-6:
            print(f"             ! the detector is only {2 * self.det_half_height_mm:.1f} mm "
                  f"tall; slices with |z| > {self.det_half_height_mm:.1f} mm have no "
                  f"data (they come out 0)")
        if not (abs(self.span_deg - 360) < 1 or abs(self.span_deg - 180) < 1):
            print(f"             ! span_deg = {self.span_deg:g}: only 180 / 360 deg are "
                  f"complete for parallel beam; FBP normalisation is extrapolated, "
                  f"result is approximate")
        return self

    def crosscheck_sidecar(self):
        """If ``<stem>_geom.json`` sits next to the file, print a comparison.

        Purely a sanity print -- a wrong span / rot_dir in the parameter cell
        is otherwise invisible until the reconstruction comes out smeared.
        Nothing here is used as an input.
        """
        if self.source is None:
            return self
        side = self.source.with_name(self.source.name.split(".nii")[0] + "_geom.json")
        if not side.exists():
            return self
        try:
            meta = json.loads(side.read_text())
            g = meta.get("geometry", {})
            checks = [("views", self.n_views, g.get("num_angle")),
                      ("span_deg", self.span_deg, g.get("angle_range_deg")),
                      ("rot_dir", self.rot_dir, g.get("rot_dir")),
                      ("det rows", self.det_rows, g.get("det_row_count")),
                      ("det cols", self.det_cols, g.get("det_col_count")),
                      ("pitch mm", self.det_pitch, g.get("det_pitch_mm"))]
            bad = [(k, a, b) for k, a, b in checks
                   if b is not None and abs(float(a) - float(b)) > 1e-6]
            r = meta.get("fov_radius_mm")
            r = None if r is None else float(r)
        except Exception as e:                          # noqa: BLE001
            print(f"cross-check: cannot read {side.name} ({e}); skipped")
            return self
        print(f"cross-check: {side.name} (checked only, not used as input) -> "
              + ("all consistent" if not bad else "**MISMATCH**"))
        for k, a, b in bad:
            print(f"             ! {k}: using {a}, file says {b}  <- fix the parameter cell")
        if r is not None:
            print(f"             FOV radius: now {self.fov_radius_mm:.2f} mm, "
                  f"at re-projection {r:.2f} mm "
                  f"(diff {self.fov_radius_mm - r:+.2f} mm)")
        return self

    # ------------------------------------------------------------------ #
    #  geometry
    # ------------------------------------------------------------------ #
    def vectors(self):
        """The (n_views, 12) ``parallel3d_vec`` array, in VOXEL units.

        Row = [rayX rayY rayZ | dX dY dZ | uX uY uZ | vX vY vZ], identical to
        ``ParallelGeometry.vectors`` in reproject_parallel.py:
        ray ``(-sin t, cos t, 0)``, detector centre on the axis, ``u`` in
        plane and ``v`` along +z.
        """
        t = np.radians(self.view_angle_deg) * self.rot_dir
        c, s = np.cos(t), np.sin(t)
        pitch = self.det_pitch / self.voxel_size

        V = np.zeros((t.size, 12))
        V[:, 0], V[:, 1] = -s, c                  # ray direction
        # V[:, 3:6] = 0 -> the detector is centred on the rotation axis
        V[:, 6], V[:, 7] = c * pitch, s * pitch   # u: in plane, +x at t = 0
        V[:, 11] = pitch                          # v: always +z (parallel!)
        return V

    def _geometries(self):
        proj_geom = astra.create_proj_geom('parallel3d_vec', self.det_rows,
                                           self.det_cols, self.vectors())
        nx, ny, nz = self.volume_shape
        vol_geom = astra.create_vol_geom(ny, nx, nz)   # (rows y, cols x, slices z)
        return proj_geom, vol_geom

    def _fbp_scale(self):
        """``pitch_u * pitch_v * dphi`` -- see the class docstring.

        ``pitch_*`` undo ASTRA's ``1/(pitch_u*pitch_v)`` backprojection weight;
        ``dphi`` is the angular integration step, halved for a 360 deg orbit
        because every line is then measured twice.
        """
        pitch = self.det_pitch / self.voxel_size
        span = float(self.span_deg)
        cover = 0.5 if abs(span - 360.0) < 1.0 else 1.0
        return pitch * pitch * np.radians(span) / self.n_views * cover

    def _filter_rows(self, window):
        """Ramp-filter every detector row, straight into the (v, view, u) layout.

        Row by row rather than all at once: the padded FFT of one row is a few
        tens of MB, of the whole stack it would be tens of GB.
        """
        n, rows, cols = self.proj.shape
        n_pad = int(2 ** np.ceil(np.log2(2 * cols)))
        H = _ramp_response(n_pad, self.det_pitch / self.voxel_size, window)
        out = np.empty((rows, n, cols), np.float32)
        for r in tqdm(range(rows), desc=f"ramp {window}", unit="row"):
            P = np.fft.rfft(self.proj[:, r, :].astype(np.float64), n=n_pad, axis=-1)
            out[r] = np.fft.irfft(P * H, n=n_pad, axis=-1)[:, :cols]
        return out

    # ------------------------------------------------------------------ #
    #  run
    # ------------------------------------------------------------------ #
    def run(self, algorithm="FBP", fbp_filter="ram-lak", iterations=300,
            fov_mask=True, gpu=None):
        """FBP or SIRT -> ``self.rec_cm``, (z, y, x) in cm^-1."""
        algorithm = str(algorithm).upper()
        fbp_filter = str(fbp_filter).lower()
        if algorithm not in ("FBP", "SIRT"):
            raise ValueError(f'algorithm must be "FBP" or "SIRT", got {algorithm!r}')
        if algorithm == "FBP":
            _ramp_response(8, 1.0, fbp_filter)          # validate the name early
        proj_geom, vol_geom = self._geometries()
        nx, ny, nz = self.volume_shape

        print(f"reconstructing {self.n_views} parallel views with {algorithm}"
              + (f" (filter {fbp_filter})" if algorithm == "FBP" else
                 f", {iterations} iterations")
              + f" -> volume {nx} x {ny} x {nz} @ {self.voxel_size:.2f} mm ...")
        t0 = time.time()
        if algorithm == "FBP":
            data = self._filter_rows(fbp_filter)
            rec = _astra_run("BP3D_CUDA", proj_geom, vol_geom, data, gpu=gpu)
            del data
            rec *= self._fbp_scale()
        else:
            rec = _astra_run("SIRT3D_CUDA", proj_geom, vol_geom,
                             self.proj.transpose(1, 0, 2), iterations, gpu)
        # voxel^-1 -> cm^-1, the same scaling as the cone version
        self.rec_cm = rec * 10.0 / self.voxel_size
        del rec
        self.label = (f"FBP_{fbp_filter}" if algorithm == "FBP"
                      else f"SIRT_it{iterations}")
        print(f"   done in {time.time() - t0:.1f} s;  mu range "
              f"[{self.rec_cm.min():.3f}, {self.rec_cm.max():.3f}] cm^-1")
        if fov_mask:
            self.apply_fov_mask()
        return self

    def apply_fov_mask(self, radius_mm=None):
        """Zero everything outside the measured cylinder and the detector's z.

        Outside ``fov_radius_mm`` the sinogram is zero by construction (the
        re-projection masked the volume there), so this only removes what the
        reconstruction itself invented -- streaks in FBP, smeared corners in
        SIRT.
        """
        if self.rec_cm is None:
            raise RuntimeError("run() first")
        R = self.fov_radius_mm if radius_mm is None else float(radius_mm)
        nz, ny, nx = self.rec_cm.shape
        vx = self.voxel_size
        x = (np.arange(nx) + 0.5 - nx / 2.0) * vx
        y = (np.arange(ny) + 0.5 - ny / 2.0) * vx
        z = (np.arange(nz) + 0.5 - nz / 2.0) * vx
        keep_xy = (x[None, :] ** 2 + y[:, None] ** 2) <= R ** 2      # (ny, nx)
        keep_z = np.abs(z) <= self.det_half_height_mm                # (nz,)
        self.rec_cm[:, ~keep_xy] = 0.0
        self.rec_cm[~keep_z] = 0.0
        print(f"FOV mask   : r <= {R:.2f} mm and |z| <= {self.det_half_height_mm:.1f} mm "
              f"-> {keep_xy.mean() * keep_z.mean() * 100:.1f}% of the voxels kept, "
              f"mu range [{self.rec_cm.min():.3f}, {self.rec_cm.max():.3f}] cm^-1")
        return self
