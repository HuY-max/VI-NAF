# VI-NAF

Neural attenuation fields (NAF) for cone-beam CT, trained directly on the measured projections with the exact ASTRA `cone_vec` geometry of the scanner. A multiresolution hash-grid implicit neural representation (tiny-cuda-nn) is fitted to the line integrals of every detector ray, and the trained network is evaluated on the same voxel grid as an ASTRA reconstruction, so the volumes overlay voxel for voxel.

Two acquisitions of the same scanner are supported:

- **CBCT**: conventional cone-beam CT with one source on the rotation axis and a half-fan (laterally offset) flat panel.
- **msCBCT**: multi-source CBCT with 8 sources along z that fire in turn, each onto its own 65-row detector band, while the gantry keeps rotating by 0.125 deg per frame.

A trained model can be re-projected onto a scanner-independent parallel-beam detector, so models from either acquisition produce directly comparable projections. The notebooks in `reconstruction/` give SIRT baselines of the measured scans and FBP / SIRT reconstructions of the re-projections, each saved as an attenuation map, an HU volume and an HU DICOM series.

## Repository layout

```
VI-NAF/
├── CBCT/                          NAF for conventional CBCT (or one msCBCT source)
│   ├── config.json                training configuration
│   ├── main.py                    entry point: python main.py [--config FILE]
│   ├── train.py                   training loop, checkpoints, volume export
│   ├── dataset.py                 ray sampler (one item = one view)
│   ├── geometry.py                layout detection from the frame size, exact rays
│   ├── params.py                  scanner calibration
│   └── reproject_parallel.py      parallel-beam re-projection of a trained model
├── msCBCT/                        NAF for multi-source CBCT
│   ├── config.json
│   ├── main.py
│   ├── train.py                   adds per-source L1 monitoring
│   ├── dataset.py                 ray sampler (one item = one frame)
│   ├── geometry.py                per-frame angle / source schedule, exact rays
│   ├── params.py                  scanner calibration
│   └── reproject_parallel.py
└── reconstruction/                ASTRA reference reconstructions (Jupyter)
    ├── CBCT_reconstruction.ipynb       SIRT of a measured CBCT scan
    ├── msCBCT_reconstruction.ipynb     SIRT of a measured msCBCT scan
    ├── parallel_reconstruction.ipynb   FBP / SIRT of a parallel re-projection
    ├── astra_recon.py                  ASTRA reconstructions used by the notebooks
    ├── hu.py                           mu -> HU conversion and DICOM export
    ├── attenuation_water_air.csv       mu of water and air vs energy, for HU
    └── params.py                       scanner calibration
```

`params.py` is the single description of the scanner (SOD / ODD, panel size, half-fan offset, source and band z tables, rotation per frame). The three copies are identical; keep them that way. Nothing geometric goes into the config files: the geometry follows from `params.py`, the projection frame size and the pixel width.

## Installation

Tested with Python 3.10 on Linux with an NVIDIA RTX 5090 and CUDA 12.8. An NVIDIA GPU is required for training (tiny-cuda-nn) and for all ASTRA algorithms.

```bash
conda create -n vinaf python=3.10
conda activate vinaf

# 1) PyTorch built for your CUDA driver
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# 2) the remaining packages
pip install -r requirements.txt

# 3) tiny-cuda-nn PyTorch bindings (compiled; needs nvcc matching torch's CUDA)
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# 4) a Jupyter frontend for reconstruction/
pip install jupyterlab

# check
python -c "import tinycudann, astra; print(astra.use_cuda())"    # -> True
```

## Data

Data are not included. Put each dataset in `VI-NAF/input/<dataset>/` (the default configs point to `../input/...`), or edit `in_dir` in the config.

The projections are the output of the scanner's preprocessing: one NIfTI file per scan.

| | CBCT | msCBCT |
|---|---|---|
| on-disk dims (v, u, frames) at 0.4 mm | (287, 372, N) | (65, 372, 8 x views) |
| frame | the full flat panel | one source's band |
| frame order | view k at k * 360 / N deg | frame k = view * 8 + s (source index s = 0..7); source s fires s * 0.125 deg after its view |

For both:

- Values are line integrals `-ln(I / I0)` (dimensionless).
- The detector pitch is stored in the header; `voxel_size` in the config must equal it and be an integer multiple of the native 0.2 mm pixel.
- Row (v) index increases with +z. For files stored the other way round, CBCT has `"flip_v": true`.
- Views are evenly spaced over 360 deg, starting at 0 deg.
- msCBCT only: the bands of sources 1 and 8 lie partly off the panel. Those rows are exactly 0 in the file; they are excluded from sampling, and a data check stops training if a sampled row is never measured.

No sensor-position or angle files are needed: all geometry comes from `params.py` and the frame size.

## CBCT workflow

1. **Configure.** Edit `CBCT/config.json`: at least `in_dir`, `proj_file` and `tag` (see [Configuration](#configuration)).
2. **Train.**
   ```bash
   cd CBCT
   python main.py                    # or: python main.py --config other.json
   ```
   The acquisition is identified from the frame size and printed; a full panel shows `z_src / z_det : +0.00 / -3.20 mm` and `row 0 -> -z (ASTRA, no flip)`. Every `save_epoch` epochs it writes
   - `model/model_<tag>_<epoch>.pkl`: network checkpoint,
   - `output/<tag>_<epoch>.nii`: mu volume [cm^-1] on the reconstruction grid,
   - `output/<tag>_loss_log.csv` and `output/<tag>_loss_curve.png`: average L1 per epoch.
3. **Re-project.** Edit `REPROJECT_CONFIG` at the bottom of `reproject_parallel.py` if needed (the defaults re-project the run's final checkpoint into `output/`), then
   ```bash
   python reproject_parallel.py      # options: --config FILE  --model CKPT.pkl  --out DIR
   ```
   See [Parallel re-projection](#parallel-re-projection).
4. **Reconstruct.** Open `reconstruction/parallel_reconstruction.ipynb`, set `PROJ_FILE` to the re-projection (`output/<tag>_<epoch>_par3d_1440.nii`) and run all cells. See [Reconstruction notebooks](#reconstruction-notebooks).

**One msCBCT source as a single-source scan.** Set `"msCBCT": true` and `"source": 1..8`, and point `in_dir` / `proj_file` at msCBCT data. `proj_file` may be the full interleaved msCBCT file; only the chosen source's frames are used. That source's z position, band and firing delay are applied automatically.

## msCBCT workflow

The same steps in `msCBCT/`:

```bash
cd msCBCT
python main.py                       # 1-2. configure config.json, then train
python reproject_parallel.py         # 3.   re-project the final checkpoint
```

then reconstruct the re-projection with `reconstruction/parallel_reconstruction.ipynb` (step 4).

One epoch visits every frame of every source once. In addition to the CBCT outputs, training writes `output/<tag>_loss_per_source.png` and logs the L1 of each source in the CSV; at every `save_epoch` it prints a per-source table (L1, L1 divided by the source's mean measurement, fraction of the band that is measured). The L1 is absolute, so compare sources by the normalised column.

## Configuration

`file` section:

| Key | Used by | Meaning |
|---|---|---|
| `in_dir` | both | folder of the projection file, relative to the run folder |
| `proj_file` | both | projection file name |
| `tag` | both | run name; prefixes every output file |
| `model_dir` | both | checkpoint folder |
| `out_dir` | both | volumes and loss logs; default output folder of `reproject_parallel.py` |
| `voxel_size` | both | [cm] projection pixel width, also the voxel size of the output grid |
| `img_size_mm` | both | [mm] (x, y, z) extent of the output grid, centred on the rotation axis |
| `rot_dir` | both | gantry rotation sense, +1 or -1 |
| `msCBCT` | CBCT | `false`: conventional CBCT; `true`: one msCBCT source; checked against the frame size (`null` skips the check) |
| `source` | CBCT | msCBCT source 1..8 (only with `"msCBCT": true`) |
| `flip_v` | CBCT | `true` if the file stores row 0 at +z |

`train` section (same keys for both):

| Key | Meaning |
|---|---|
| `gpu` | CUDA device index |
| `lr` | Adam learning rate |
| `epoch` | number of epochs; one epoch = every view (CBCT) or frame (msCBCT) once |
| `save_epoch` | write a checkpoint and a volume every this many epochs |
| `num_sample_ray` | rays drawn per view / frame |
| `batch_size` | views / frames per gradient step |
| `lr_decay_epoch`, `lr_decay_coefficient` | step learning-rate decay |
| `grid_chunk_size` | z slices per chunk when the volume is evaluated |
| `infer_chunk_size` | points per network call when the volume is evaluated |

`encoding` and `network` are passed unchanged to tiny-cuda-nn (`NetworkWithInputEncoding`): a 16-level hash grid and a fully fused MLP. The loss is the L1 distance between measured and predicted line integrals; each ray is sampled at one-voxel steps over twice the source-to-isocentre distance.

## Parallel re-projection

`reproject_parallel.py` evaluates a trained network on the output grid and projects it with ASTRA's `parallel3d_vec` onto a full, centred detector. A parallel beam has no source, magnification, cone angle or detector offset, so the result depends only on the volume and the view angle, and models trained on either acquisition land on the same detector.

`REPROJECT_CONFIG`:

| Key | Default | Meaning |
|---|---|---|
| `model_path` | `None` | checkpoint; `None` = `<model_dir>/model_<tag>_<train.epoch>.pkl` |
| `num_angle` | 1440 | views evenly over 360 deg |
| `fov_margin_xy` | 1.2 | scales the radius of the FOV cylinder |
| `fov_margin_z` | 1.05 | scales its height |
| `out_path` | `None` | output folder; `None` = the config's `out_dir` |
| `out_name` | `None` | file stem; `None` = checkpoint name without `model_`, plus `_par3d_<num_angle>` |
| `angle_chunk` | 120 | views per ASTRA call (GPU memory) |
| `save_volume` | `False` | also write the masked mu volume |
| `preview` | `True` | write a preview image |
| `verify_sirt` | 0 | if > 0: reconstruct the result with that many SIRT iterations and report the error |
| `gpu` | `None` | `None` = the config's `train.gpu` |

- **Detector**: the half-fan panel mirrored about the axis, 726 x 287 px = 290.4 x 114.8 mm at 0.4 mm; the lateral offset is undone and the height is unchanged.
- **FOV mask**: before projecting, the volume is zeroed outside the measured cylinder, because the network is unconstrained there. Its radius is `R = SOD * sin(atan(u_far / SDD)) = 93.3 mm` (the panel's outer ray, rebinned to parallel) times `fov_margin_xy`. Its z range is the measured axial coverage at the rotation axis, scaled about its centre by `fov_margin_z`: [-39.94, +35.71] mm for a full CBCT panel, [-51.56, +52.39] mm for the union of the 8 msCBCT bands, about 17 mm for one complete band.

Outputs in `out_path`:

- `<out_name>.nii`: dims (v, u, angle) = (287, 726, 1440), dimensionless line integrals, pitch in the header (the same layout as the measured projections);
- `<out_name>_geom.json`: the output geometry, view angles and FOV;
- `<out_name>_preview.png` (with `preview`);
- `<out_name>_volume.nii` (with `save_volume`) and `<out_name>_sirt.nii` (with `verify_sirt`).

## Reconstruction notebooks

Run the notebooks from `reconstruction/` and edit only the parameter cell (section 2). The ASTRA code lives in `astra_recon.py`, the HU conversion and DICOM export in `hu.py`.

- `CBCT_reconstruction.ipynb`: SIRT of a measured CBCT scan on the exact `cone_vec` geometry used for NAF training (including the panel's vertical offset).
- `msCBCT_reconstruction.ipynb`: SIRT of a measured msCBCT scan, with each frame's source, band and angle.
- `parallel_reconstruction.ipynb`: FBP or SIRT of a re-projection from `reproject_parallel.py`. For a parallel beam each detector row is an independent 2D sinogram, so FBP is exact (no cone or redundancy weighting). The `_geom.json` next to the file is only cross-checked, never used as input.

The geometry comes from `params.py` and the file header; only the angular span, start angle and rotation sense are set in the parameter cell and must match the scan.

Each notebook writes three outputs to `reconstruction/output/`, with `<stem> = recon_<name>_<algorithm>` and `<name>` the input file name:

| File | Content |
|---|---|
| `<stem>.nii` | linear attenuation coefficient mu [cm^-1] |
| `<stem>_HU.nii` | Hounsfield units |
| `<stem>_HU_dicom/` | the HU volume as a CT DICOM series (one file per axial slice, int16, stored value = HU) |

HU are computed at the mean energy of the beam, `MEAN_ENERGY_KEV` in the parameter cell:

```
HU = (mu - mu_water) / (mu_water - mu_air) * 1000
```

with mu_water and mu_air from `attenuation_water_air.csv` (0.5 keV steps, linearly interpolated). The mean energies are **95 keV for CBCT** and **67 keV for msCBCT**; for a parallel re-projection, use the energy of the scan the model was trained on. The DICOM series carries the phantom name (`PHANTOM`; default: the input file name), the device, kVp (`KVP`, default 110) and tube current (`TUBE_CURRENT_MA`, default 11 mA); the energy and the mu_water / mu_air used are recorded in `ImageComments`.

## Outputs and conventions

- NAF volumes and `<stem>.nii` are linear attenuation coefficients mu in cm^-1; `<stem>_HU.nii` holds HU. All are float32 NIfTI with dims (x, y, z) and the voxel spacing in mm in the header.
- DICOM slices use `ImageOrientationPatient = [1, 0, 0, 0, 1, 0]` and the same centred grid as the NIfTI files.
- The rotation axis passes through the centre of the volume; +z runs along the axis. At gantry angle 0 the source is at -y and detector columns run along +x (ASTRA's volume frame).
- NAF volumes, SIRT volumes and parallel-beam reconstructions on the default grid (240 x 240 x 120 mm at 0.4 mm) share the same voxel grid.

## License

Released under the [MIT License](LICENSE).
