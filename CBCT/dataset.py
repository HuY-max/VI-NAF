"""dataset.py — PyTorch Dataset for NAF training on SINGLE-SOURCE cone-beam
CT with exact ASTRA cone_vec ray geometry (see geometry.py), with the same
frame-level sampling as ../msCBCT/dataset.py.

One __getitem__ = one projection VIEW, with ``num_sample_ray`` rays drawn
from that view's valid detector pixels.  With a single source a view IS a
frame, so this is both the classic CBCT NAF sampling and the frame-level
sampling of the msCBCT code — no distinction to make.

Projection file (SimpleITK numpy view): (num_views, cols u, rows v).
Row order on load is normalised to the ASTRA convention (r increases with
+z) using layout.flip_v — see geometry.py's module docstring.
"""

import numpy as np
import SimpleITK as sitk
from torch.utils import data

import geometry as geo
from params import BinnedGeometry


class TrainData(data.Dataset):

    def __init__(self, proj_path, geom: BinnedGeometry, num_sample_ray,
                 source=None, rot_dir=+1, flip_v=False, is_mscbct=None,
                 orbit_deg=360.0):
        self.geom = geom
        self.num_sample_ray = num_sample_ray
        self.rot_dir = rot_dir

        # ---- Projection data ----
        proj = sitk.GetArrayFromImage(sitk.ReadImage(proj_path))
        print(f"Projection file (num_frames, cols u, rows v): {proj.shape}")

        # ---- Geometry from the frame size ----
        self.layout = geo.detect_layout(proj.shape[1:], geom, source=source,
                                        flip_v=flip_v, is_mscbct=is_mscbct)
        self.layout.summary()
        self.num_det = self.layout.num_det

        # A band layout may be pointed at the FULL interleaved msCBCT file;
        # keep only this source's frames if so.
        if self.layout.kind == "band":
            proj, was_interleaved = geo.deinterleave_if_needed(
                proj, self.layout.source, geom.params.source_number)
            if was_interleaved:
                print(f"  interleaved msCBCT file detected -> kept source "
                      f"{self.layout.source + 1} only: {proj.shape[0]} views")
        self.num_views = proj.shape[0]

        if self.layout.flip_v:                       # file row 0 at +z -> ASTRA
            proj = proj[:, :, ::-1]

        # (view, c, r) -> (r, c, view) -> (num_det, num_views); index r*cols + c
        self.proj = np.ascontiguousarray(
            proj.transpose(2, 1, 0)).reshape(self.num_det, self.num_views)
        del proj

        # ---- View schedule ----
        self.angle_deg = geo.view_angles(self.num_views, self.layout,
                                         orbit_deg=orbit_deg)
        print(f"views: {self.num_views} over {orbit_deg} deg, step "
              f"{orbit_deg / self.num_views:.4f} deg")

        # ---- Exact rays at gantry 0 ----
        R = geom.params.SOD / geom.pixel_size          # normalization [voxels]
        self.R = R
        origin, dirs, valid = geo.rays_zero_deg(self.layout, geom)
        self.origin_norm = (origin / R).astype(np.float32)     # (3,)
        self.dirs = dirs                                       # (num_det, 3)

        self.pool = np.flatnonzero(valid)
        if self.pool.size < num_sample_ray:
            raise ValueError(f"valid pixel pool ({self.pool.size}) < "
                             f"num_sample_ray ({num_sample_ray})")

        # ---- A band's pool must not contain padding ----
        # A band row that is exactly 0 in every view is a row the panel never
        # covered (sources 1 / 8); sampling it asserts "no attenuation" along a
        # ray through material.  Checked against the data, so a wrong
        # valid-row range cannot pass silently.
        if self.layout.kind == "band":
            rows3 = self.proj.reshape(self.layout.rows, self.layout.cols,
                                      self.num_views)
            dead = np.flatnonzero(~np.any(rows3, axis=(1, 2)))
            bad = np.intersect1d(dead, np.unique(self.pool // self.layout.cols))
            if bad.size:
                raise ValueError(
                    f"source {self.layout.source + 1}: band row(s) {bad.tolist()} "
                    f"are exactly 0 in all {self.num_views} views (never covered "
                    f"by the panel) but are in the sampling pool — the "
                    f"valid-row range is wrong for this source.")

        self.num_samples = int(2 * round(R))
        self.y_rel = geo.sample_offsets(self.num_samples)
        print(f"R = SOD/pixel = {R:.1f} voxels -> {self.num_samples} samples/ray "
              f"(step {2 * R / (self.num_samples - 1):.4f} voxel)")

    def __getitem__(self, item):
        """Rays + measured line integrals for view `item`.

        Returns
        -------
        ray_sample  : (num_sample_ray, num_samples, 3) float32 — normalized
                      sample coordinates, rotated to this view's gantry angle.
        proj_sample : (num_sample_ray,) float32 — the rays' measurements.
        """
        det_idx = self.pool[np.random.choice(self.pool.size,
                                             size=self.num_sample_ray,
                                             replace=False)]

        # points at gantry 0: origin + t * dir   (normalized units)
        d = self.dirs[det_idx]                                # (nray, 3)
        pts = (self.origin_norm[None, None, :]
               + d[:, None, :] * self.y_rel[None, :, None])   # (nray, nsamp, 3)

        pts = geo.rotate_z(pts, self.angle_deg[item], self.rot_dir)

        proj_sample = self.proj[det_idx, item]
        return pts, proj_sample

    def __len__(self):
        return self.num_views
