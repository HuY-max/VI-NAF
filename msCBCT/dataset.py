"""dataset.py — PyTorch Dataset for NAF training on multi-source CBCT with
EXACT ASTRA cone_vec ray geometry (see geometry.py).

One __getitem__ = ONE FRAME (one source firing at one gantry angle) and
__len__ = num_proj (2880 for the full real scan).  The DataLoader shuffles
the frames directly, so over an epoch EVERY frame of EVERY source is visited
exactly once — per-source coverage is uniform by construction.

Because all rays of an item belong to a single frame, they share one gantry
angle — the per-frame continuous-rotation angle (view * view_step +
src * 0.125 deg) is still exact, it is just applied per item instead of
per ray.

Projection file (SimpleITK numpy view): (num_proj, cols u, H v), frame
k = view * num_src + src (firing order).  Row r increases with +z, column c
increases with +x at gantry 0 — the ASTRA conventions the file was written in.
"""

import numpy as np
import SimpleITK as sitk
from torch.utils import data

import geometry as geo
from params import BinnedGeometry


class TrainData(data.Dataset):
    """One __getitem__ = one FRAME (source k % num_src at frame k's angle),
    with ``num_sample_ray`` rays drawn from that source's valid pixel pool.
    """

    def __init__(self, proj_path, geom: BinnedGeometry, num_sample_ray,
                 rot_dir=+1, intra_view_stepping=True, orbit_deg=360.0):
        self.geom = geom
        self.num_sample_ray = num_sample_ray
        self.rot_dir = rot_dir
        self.num_src = geom.params.source_number

        # ---- Projection data ----
        proj = sitk.GetArrayFromImage(sitk.ReadImage(proj_path))
        print(f"Projection file (num_proj, cols u, H v): {proj.shape}")
        self.num_proj = proj.shape[0]
        if proj.shape[1:] != (geom.cols, geom.H):
            raise ValueError(
                f"projection frames are {proj.shape[1:]}, geometry expects "
                f"(cols={geom.cols}, H={geom.H}) — wrong voxel_size for this "
                f"projection file?")
        self.num_det = geom.H * geom.cols
        self.num_angle = self.num_proj // self.num_src

        # (k, c, r) -> (r, c, k) -> (num_det, num_proj); det index = r*cols + c
        self.proj = np.ascontiguousarray(
            proj.transpose(2, 1, 0)).reshape(self.num_det, self.num_proj)
        del proj

        # ---- Per-frame schedule ----
        self.frame_angle_deg, self.frame_source = geo.frame_schedule(
            self.num_proj, geom, orbit_deg=orbit_deg,
            intra_view_stepping=intra_view_stepping)
        print(f"frames: {self.num_proj} = {self.num_angle} views x "
              f"{self.num_src} sources, view step "
              f"{orbit_deg / self.num_angle:.4f} deg, intra-view stepping "
              f"{'ON (+%.3f deg/frame)' % geom.params.deg_per_frame if intra_view_stepping else 'OFF'}")
        print(f"sampling unit: ONE FRAME -> dataset length {self.num_proj} "
              f"(frame-level, every source visited every epoch)")

        # ---- Exact rays at gantry 0 ----
        R = geom.params.SOD / geom.pixel_size          # normalization [voxels]
        self.R = R
        origins, dirs, valid = geo.rays_zero_deg(geom)
        self.origins_norm = (origins / R).astype(np.float32)   # (num_src, 3)
        self.dirs = dirs                                       # (num_src, num_det, 3)

        # ---- Per-source pixel pools ----
        # never-covered rows of sources 1/8 hold zeros, not measurements
        self.pools = [np.flatnonzero(valid[s]) for s in range(self.num_src)]
        n_missing = valid.size - sum(p.size for p in self.pools)
        print("sampling pools per source: "
              + ", ".join(f"s{s + 1}:{p.size}" for s, p in enumerate(self.pools))
              + f" — {n_missing} never-measured pixels masked "
              f"(partial bands of sources 1/8)")
        for s, p in enumerate(self.pools):
            if p.size < num_sample_ray:
                raise ValueError(
                    f"source {s + 1} pool ({p.size}) < num_sample_ray "
                    f"({num_sample_ray})")

        # ---- The pools must not contain padding ----
        # A band row that is exactly 0 in EVERY frame of its source is a row
        # the panel never covered.  Sampling one teaches the network "no
        # attenuation along this ray", which no volume can satisfy.  Checked
        # against the data itself, so a wrong row range cannot pass silently.
        rows3 = self.proj.reshape(geom.H, geom.cols, self.num_proj)
        for s in range(self.num_src):
            dead = np.flatnonzero(~np.any(rows3[:, :, s::self.num_src], axis=(1, 2)))
            bad = np.intersect1d(dead, np.unique(self.pools[s] // geom.cols))
            if bad.size:
                raise ValueError(
                    f"source {s + 1}: band row(s) {bad.tolist()} are exactly 0 "
                    f"in all {self.num_angle} frames (never covered by the "
                    f"panel) but are in the sampling pool — the valid-row "
                    f"range is wrong for this source.")

        # ---- Per-source measured level ----
        # The per-source L1 the trainer logs is an ABSOLUTE residual, so a
        # source whose rays cross more material shows a larger L1 at equal
        # relative accuracy.  These levels are what to divide by before
        # calling one source "worse than" another.
        self.src_level = np.array(
            [float(np.abs(self.proj[np.ix_(self.pools[s],
                                          np.flatnonzero(self.frame_source == s))]
                          ).mean())
             for s in range(self.num_src)])
        print("mean |measurement| per source (valid pixels): "
              + ", ".join(f"s{s + 1}:{v:.4f}"
                          for s, v in enumerate(self.src_level)))

        self.num_samples = int(2 * round(R))
        self.y_rel = geo.sample_offsets(self.num_samples)
        print(f"R = SOD/pixel = {R:.1f} voxels -> {self.num_samples} samples/ray "
              f"(step {2 * R / (self.num_samples - 1):.4f} voxel)")

    def __getitem__(self, item):
        """Rays + measured line integrals for frame `item`.

        Returns
        -------
        ray_sample  : (num_sample_ray, num_samples, 3) float32 — normalized
                      sample coordinates, rotated by this frame's gantry angle.
        proj_sample : (num_sample_ray,) float32 — the rays' measurements.
        src         : int64 — 0-based index of the source that fired this
                      frame.  Used ONLY to split the logged L1 per source
                      (see train.py); it never enters the objective, which
                      stays the plain L1 over all rays.
        """
        src = int(self.frame_source[item])
        pool = self.pools[src]
        det_idx = pool[np.random.choice(pool.size,
                                        size=self.num_sample_ray,
                                        replace=False)]

        # points at gantry 0: origin + t * dir   (normalized units)
        d = self.dirs[src, det_idx]                           # (nray, 3)
        pts = (self.origins_norm[src][None, None, :]
               + d[:, None, :] * self.y_rel[None, :, None])   # (nray, nsamp, 3)

        # rotate by this frame's continuous-rotation angle
        pts = geo.rotate_z(pts, self.frame_angle_deg[item], self.rot_dir)

        proj_sample = self.proj[det_idx, item]
        return pts, proj_sample, src

    def __len__(self):
        return self.num_proj
