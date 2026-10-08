"""train.py — NAF/INR training loop for multi-source CBCT with the exact
ASTRA cone_vec geometry (see geometry.py / dataset.py).

The dataset is FRAME-level (one item = one of the num_proj frames, e.g.
2880), so one epoch = one full shuffled pass over ALL frames of ALL sources
(len(loader) = ceil(num_proj / batch_size) gradient steps).

Because one item is one frame, the source that fired it is known, so the L1
is also logged PER SOURCE (no_grad, monitoring only — the objective stays the
plain L1 over all rays).  Sources 1 and 8, whose bands are partly off the
panel, typically dominate it; divide by the per-source measurement level
before ranking sources.

Outputs (``tag`` from the config):
  <model_dir>/model_<tag>_<epoch>.pkl   network checkpoint, every save_epoch
  <out_dir>/<tag>_<epoch>.nii           mu [cm^-1] on the ASTRA grid
  <out_dir>/<tag>_loss_log.csv          average L1 per epoch, total + per source
  <out_dir>/<tag>_loss_curve.png        total L1 curve
  <out_dir>/<tag>_loss_per_source.png   per-source curves + last-epoch bars
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
    # DERIVED from it — the band calibration tables in SystemParams live on
    # the native grid, so BinnedGeometry still needs the ratio internally.
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
        rot_dir=rot_dir)
    num_samples = train_set.num_samples
    num_src = train_set.num_src
    src_level = train_set.src_level                # mean |measurement|, per source
    pool_frac = np.array([p.size for p in train_set.pools]) / train_set.num_det
    train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size,
                                   shuffle=True, num_workers=4, pin_memory=True)
    rays_per_epoch = len(train_set) * num_sample_ray
    print(f"epoch = {len(train_loader)} steps of {batch_size} frames x "
          f"{num_sample_ray} rays = {rays_per_epoch} rays")

    grid_chunk_z = cfg_t.get("grid_chunk_size", 16)
    infer_chunk_size = cfg_t.get("infer_chunk_size", 500000)

    # ---- optimization & reconstruction ------------------------------------
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    epoch_loss_hist = []
    src_loss_hist = []                             # per epoch: (num_src,) L1

    loop_tqdm = tqdm(range(epoch), leave=False)
    for e in loop_tqdm:
        network.train()
        loss_log = 0
        src_err = np.zeros(num_src)                # sum of per-frame L1
        src_cnt = np.zeros(num_src)                # frames seen, per source
        for i, (ray, proj, src) in enumerate(train_loader):
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

            # per-frame L1, split by the source that fired the frame.  The
            # objective above is untouched — this is monitoring only.
            with torch.no_grad():
                frame_l1 = (proj_pre - proj).abs().mean(dim=1).cpu().numpy()
                s_np = src.numpy()
                src_err += np.bincount(s_np, weights=frame_l1, minlength=num_src)
                src_cnt += np.bincount(s_np, minlength=num_src)

        avg_loss = loss_log / max(1, len(train_loader))
        epoch_loss_hist.append(avg_loss)
        src_loss_hist.append(src_err / np.maximum(src_cnt, 1))

        scheduler.step()
        loop_tqdm.set_description(tag)
        # 8 per-source numbers do not fit the progress bar — show the extremes
        s_l1 = src_loss_hist[-1]
        lo, hi = int(np.argmin(s_l1)), int(np.argmax(s_l1))
        loop_tqdm.set_postfix(lr=scheduler.get_last_lr()[0], loss=avg_loss,
                              best=f"s{lo + 1}:{s_l1[lo]:.4f}",
                              worst=f"s{hi + 1}:{s_l1[hi]:.4f}")

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

            # —— loss history to CSV (total + per source) ——
            src_hist = np.asarray(src_loss_hist)             # (n_epochs, num_src)
            with open(f"{out_path}/{tag}_loss_log.csv", "w", newline="") as fcsv:
                w = csv.writer(fcsv)
                w.writerow(["epoch", "avg_loss"]
                           + [f"l1_src{s + 1}" for s in range(num_src)])
                for ep_idx, loss_val in enumerate(epoch_loss_hist, 1):
                    w.writerow([ep_idx, loss_val] + list(src_hist[ep_idx - 1]))

            # —— per-source table of the epoch just finished ——
            # L1 is absolute, so the level-normalized column is the fair
            # "which source fits worst" comparison; the pool column shows
            # which sources have masked (never-measured) band rows.
            s_l1 = src_hist[-1]
            print(f"  per-source L1 at epoch {e + 1} "
                  f"(total {epoch_loss_hist[-1]:.5f}):")
            print("    src   L1        L1/level   level    valid pool")
            for s in range(num_src):
                flag = "  <- masked rows" if pool_frac[s] < 0.999 else ""
                print(f"    s{s + 1}    {s_l1[s]:.5f}   {s_l1[s] / src_level[s]:6.3f}   "
                      f"{src_level[s]:.4f}   {pool_frac[s] * 100:5.1f}%{flag}")

            # —— loss curve (total) ——
            plt.figure()
            xs = np.arange(1, len(epoch_loss_hist) + 1)
            plt.semilogy(xs, epoch_loss_hist, linewidth=2)
            plt.xlabel("Epoch"); plt.ylabel("Average training loss")
            plt.title("Loss Curve"); plt.grid(True); plt.tight_layout()
            plt.savefig(f"{out_path}/{tag}_loss_curve.png", dpi=200)
            plt.close()

            # —— per-source curves + final-epoch bars ——
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
            colors = plt.cm.viridis(np.linspace(0, 0.92, num_src))
            for s in range(num_src):
                ax1.semilogy(xs, src_hist[:, s], linewidth=1.2,
                             color=colors[s], label=f"s{s + 1}")
            ax1.semilogy(xs, epoch_loss_hist, linewidth=2.2, color="k",
                         label="all sources")
            ax1.set_xlabel("Epoch"); ax1.set_ylabel("Average L1")
            ax1.set_title("L1 per source"); ax1.grid(True)
            ax1.legend(ncol=3, fontsize=8)

            bars = ax2.bar(np.arange(1, num_src + 1), s_l1, color=colors)
            for s in range(num_src):
                if pool_frac[s] < 0.999:            # sources 1 / 8: masked rows
                    bars[s].set_hatch("//")
                    bars[s].set_edgecolor("k")
            ax2.axhline(epoch_loss_hist[-1], color="k", linestyle="--",
                        linewidth=1, label="all sources")
            ax2.set_xlabel("Source"); ax2.set_ylabel("Average L1")
            ax2.set_title(f"L1 per source, epoch {e + 1} "
                          f"(hatched = masked band rows)")
            ax2.grid(True, axis="y"); ax2.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(f"{out_path}/{tag}_loss_per_source.png", dpi=200)
            plt.close(fig)
