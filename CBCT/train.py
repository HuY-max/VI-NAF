"""train.py — NAF/INR training loop for SINGLE-SOURCE cone-beam CT on the
exact ASTRA cone_vec geometry (see geometry.py / dataset.py).

The data are either a conventional CBCT scan (full flat panel, on-axis
source) or ONE source of the msCBCT scan (its 65-row band).  The layout is
derived from the projection FRAME SIZE — no mode switch; see geometry.py.

Same forward model, network and reconstruction grid as ../msCBCT/train.py,
so a volume from this code overlays the msCBCT volumes and the ASTRA SIRT
references voxel-for-voxel.

Outputs (``tag`` from the config):
  <model_dir>/model_<tag>_<epoch>.pkl   network checkpoint, every save_epoch
  <out_dir>/<tag>_<epoch>.nii           mu [cm^-1] on the ASTRA grid
  <out_dir>/<tag>_loss_log.csv          average L1 per epoch
  <out_dir>/<tag>_loss_curve.png        the same, plotted
"""

import os
import csv

import numpy as np
import torch
import SimpleITK as sitk
import tinycudann as tcnn
from tqdm import tqdm
from torch.utils import data
from torch.optim import lr_scheduler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import geometry as geo
import dataset
from params import SystemParams, BinnedGeometry


def train(config):

    # ---- files & geometry -------------------------------------------------
    cfg_f = config["file"]
    tag = cfg_f["tag"]
    in_path, out_path, model_path = cfg_f["in_dir"], cfg_f["out_dir"], cfg_f["model_dir"]
    proj_path = "{}/{}".format(in_path, cfg_f["proj_file"])

    # voxel_size [cm] = the projection pixel width (0.04 -> 0.4 mm data).
    # The internal bin factor relative to the native 0.2 mm panel grid is
    # DERIVED from it — the panel/band tables in SystemParams live on the
    # native grid, so BinnedGeometry still needs the ratio internally.
    P = SystemParams()
    voxel_mm = cfg_f["voxel_size"] * 10.0
    geom = BinnedGeometry(P, voxel_mm / P.pixel_size)
    geom.summary()
    voxel_size = geom.pixel_size / 10.0            # [cm], for Beer-Lambert
    rot_dir = cfg_f.get("rot_dir", +1)
    img_size_mm = cfg_f["img_size_mm"]
    nx, ny, nz = geo.volume_shape(img_size_mm, geom.pixel_size)
    R = geom.params.SOD / geom.pixel_size          # normalization [voxels]
    print(f"ASTRA-aligned output grid: {nx} x {ny} x {nz} voxels "
          f"@ {geom.pixel_size:.2f} mm (FOV {img_size_mm} mm), R = {R:.1f}")

    # ---- training hyper-parameters ---------------------------------------
    cfg_t = config["train"]
    lr = cfg_t["lr"]
    gpu = cfg_t["gpu"]
    epoch = cfg_t["epoch"]
    save_epoch = cfg_t["save_epoch"]
    lr_decay_epoch = cfg_t["lr_decay_epoch"]
    lr_decay_coefficient = cfg_t["lr_decay_coefficient"]
    batch_size = cfg_t["batch_size"]
    num_sample_ray = cfg_t["num_sample_ray"]

    device = torch.device(f"cuda:{gpu}")

    # ---- model ------------------------------------------------------------
    dc_loss = torch.nn.L1Loss().to(device)
    network = tcnn.NetworkWithInputEncoding(
        n_input_dims=3, n_output_dims=1,
        encoding_config=config["encoding"], network_config=config["network"]).to(device)
    optimizer = torch.optim.Adam(params=network.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_epoch,
                                    gamma=lr_decay_coefficient)

    # ---- data loader ------------------------------------------------------
    train_set = dataset.TrainData(
        proj_path=proj_path, geom=geom, num_sample_ray=num_sample_ray,
        source=cfg_f.get("source"), rot_dir=rot_dir,
        flip_v=cfg_f.get("flip_v", False),
        is_mscbct=cfg_f.get("msCBCT"))
    num_samples = train_set.num_samples
    train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size,
                                   shuffle=True, num_workers=4, pin_memory=True)
    rays_per_epoch = len(train_set) * num_sample_ray
    print(f"epoch = {len(train_loader)} steps of {batch_size} views x "
          f"{num_sample_ray} rays = {rays_per_epoch} rays")

    z0_mm, z1_mm = geo.z_coverage_mm(train_set.layout, geom)
    print(f"measured z at the rotation axis: {z0_mm:+.1f} .. {z1_mm:+.1f} mm "
          f"({z1_mm - z0_mm:.1f} mm); volume spans +/-{img_size_mm[2] / 2:.1f} mm")
    if img_size_mm[2] > (z1_mm - z0_mm) * 1.05:
        print(f"  NOTE: the volume is taller than this source measures — "
              f"voxels outside the range above are unconstrained by the data "
              f"(kept anyway so the grid overlays the msCBCT/ASTRA volumes).")

    grid_chunk_z = cfg_t.get("grid_chunk_size", 16)
    infer_chunk_size = cfg_t.get("infer_chunk_size", 500000)

    # ---- optimization & reconstruction ------------------------------------
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    loss_hist = []

    loop_tqdm = tqdm(range(epoch), leave=False)
    for e in loop_tqdm:
        network.train()
        loss_log = 0
        for i, (ray, proj) in enumerate(train_loader):
            ray = ray.to(device).float().view(-1, 3)
            proj = proj.to(device).float()

            mu_pre = network(ray).view(-1, num_sample_ray, num_samples).float()

            # Beer-Lambert: sample step = 1 voxel = voxel_size cm
            proj_pre = voxel_size * torch.sum(mu_pre, dim=2)

            proj = proj.to(proj_pre.dtype)
            loss = dc_loss(proj_pre, proj)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_log = loss_log + loss.item()

        avg_loss = loss_log / max(1, len(train_loader))
        loss_hist.append(avg_loss)

        scheduler.step()
        loop_tqdm.set_description(tag)
        loop_tqdm.set_postfix(lr=scheduler.get_last_lr()[0], loss=avg_loss)

        # ---- model save & ASTRA-grid reconstruction ----
        if (e + 1) % save_epoch == 0:
            with torch.no_grad():
                torch.save(network.state_dict(),
                           f"{model_path}/model_{tag}_{e + 1}.pkl")
                network.eval()

                # volume array [iz, iy, ix] — the ASTRA python layout
                img_pre = np.zeros((nz, ny, nx), dtype=np.float32)
                total_points = nx * ny * nz
                print(f"  Reconstructing ASTRA grid ({nx}, {ny}, {nz}) = "
                      f"{total_points:,} points")

                for xyz_chunk, z0, z1 in geo.astra_grid_chunks(
                        nx, ny, nz, R_norm=R, chunk_z=grid_chunk_z):
                    xyz_tensor = torch.from_numpy(xyz_chunk).to(device)
                    num_points = xyz_tensor.shape[0]
                    results = []
                    for i0 in range(0, num_points, infer_chunk_size):
                        i1 = min(i0 + infer_chunk_size, num_points)
                        out = network(xyz_tensor[i0:i1])[:, 0]
                        results.append(out.cpu())
                    img_pre[z0:z1] = torch.cat(results, dim=0).numpy() \
                        .reshape(z1 - z0, ny, nx)

                network.train()

            # numpy [iz, iy, ix] -> NIfTI dims (nx, ny, nz): identical on-disk
            # layout to the ASTRA SIRT reference volumes.
            img_sitk = sitk.GetImageFromArray(img_pre)
            img_sitk.SetSpacing([geom.pixel_size] * 3)
            sitk.WriteImage(img_sitk, f"{out_path}/{tag}_{e + 1}.nii")

            # —— loss history to CSV ——
            with open(f"{out_path}/{tag}_loss_log.csv", "w", newline="") as fcsv:
                w = csv.writer(fcsv)
                w.writerow(["epoch", "avg_loss"])
                for ep_idx, loss_val in enumerate(loss_hist, 1):
                    w.writerow([ep_idx, loss_val])

            # —— loss curve ——
            plt.figure()
            xs = np.arange(1, len(loss_hist) + 1)
            plt.semilogy(xs, loss_hist, linewidth=2)
            plt.xlabel("Epoch"); plt.ylabel("Average training loss")
            plt.title("Loss Curve"); plt.grid(True); plt.tight_layout()
            plt.savefig(f"{out_path}/{tag}_loss_curve.png", dpi=200)
            plt.close()
