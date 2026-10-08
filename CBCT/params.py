"""Fixed system description of the real scanner + binned-grid geometry.

SINGLE SOURCE OF TRUTH for the scanner geometry — do NOT duplicate these
numbers into config files.  Shipped identically in CBCT/, msCBCT/ and
reconstruction/; keep the copies identical.

All numbers replicate the scanner's calibrated preprocessing
(msCBCT_preprocessing_ASRS.m / msCBCT_ASTRA_geom3D.m):

* raw frames are flipped vertically at load; every row table below refers to
  the FLIPPED frame, where the row index increases towards -z;
* frame i (1-based) is fired by source ``(i-1) % 8 + 1`` (firing order);
* the measured z tables carry a +1.5 mm global calibration shift.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SystemParams:
    """Native (0.2 mm grid) scanner constants.  Defaults = the real system."""

    # ---- Flat panel (flipped raw frames) ----
    det_row_number: int = 574        # panel rows
    det_col_number: int = 744        # panel cols
    pixel_size: float = 0.2          # [mm] native detector pitch

    # ---- System geometry ----
    SOD: float = 410.0               # [mm] source -> isocentre
    ODD: float = 210.0               # [mm] isocentre -> detector
    offset_u: float = 70.5           # [mm] horizontal half-fan detector offset
    offset_v: float = 0.0            # [mm]

    # ---- Conventional CBCT (one on-axis source, full panel) ----
    cbct_offset_v: float = -3.2      # [mm] vertical detector offset of that scan

    # ---- Multi-source layout (FIRING order: frame 1 -> source 1) ----
    source_number: int = 8
    beam_width: int = 130            # [raw rows] extracted band height / source
    # Calibrated beam-centre rows (1-based, flipped frame):
    beam_center: tuple = (554, 468, 405, 335, 267, 199, 134, 55)
    # Measured z at the isocentre frame [mm] (Nuray), firing order:
    z_src_mm: tuple = (-44.58, -32.90, -21.10, -9.20, 2.80, 14.80, 26.83, 38.56)
    z_det_mm: tuple = (-53.40, -36.20, -23.60, -9.60, 4.00, 17.60, 30.60, 46.40)
    z_shift_mm: float = 1.5          # global calibration shift

    # ---- Orbit ----
    deg_per_frame: float = 0.125     # [deg] continuous gantry rotation / frame

    @property
    def SDD(self) -> float:
        return self.SOD + self.ODD

    def band_tables_raw(self):
        """Per-source band placement on the RAW grid (0-based).

        Returns ``(band_start, data)`` where band-local row ``j`` sits at panel
        row ``band_start[s] + j`` and ``data[s] = (j0, j1)`` is the half-open
        band-local row range the physical panel covers.  Reproduces the MATLAB
        det_rows/data_rows tables including the manual edge-source overrides.
        """
        n, H = self.source_number, self.beam_width
        band_start = np.array([bc - 1 - H // 2 for bc in self.beam_center])
        data = np.tile([0, H], (n, 1))
        data[0] = [0, 86]        # source 1: rows 87:130 are off the panel top
        band_start[7] = -10      # source 8: the calibrated table pins band row 11
        data[7] = [10, H]        # to panel row 1 -> rows 1:10 are off the panel
        return band_start, data


class BinnedGeometry:
    """Every grid-dependent quantity for an integer bin factor ``b``.

    The pipeline bins the raw frames FIRST (frames become ``rows x cols`` at
    ``pixel_size`` mm) and all downstream row tables live on that binned grid.
    For ``b = 1`` everything reduces exactly to the validated MATLAB tables.
    """

    def __init__(self, params: SystemParams, bin_factor: int = 1):
        b = int(round(bin_factor))
        if b < 1 or abs(b - bin_factor) > 1e-9:
            raise ValueError(f"bin factor must be a positive integer, got {bin_factor}")
        P = self.params = params
        self.b = b

        # ---- Panel on the binned grid (trailing rows/cols are cropped) ----
        self.pixel_size = P.pixel_size * b
        self.rows = P.det_row_number // b
        self.cols = P.det_col_number // b
        crop_cols = P.det_col_number - self.cols * b
        crop_rows = P.det_row_number - self.rows * b
        # Cols are cropped at high u -> shift of the kept-array centre:
        self.offset_u = P.offset_u - crop_cols / 2 * P.pixel_size
        self.offset_v = P.offset_v
        if crop_rows or crop_cols:
            print(f"[BinnedGeometry] warning: bin {b} crops {crop_rows} raw row(s) / "
                  f"{crop_cols} raw col(s) of the panel (offset_u compensated)")

        # ---- Band placement ----
        self.H = P.beam_width // b                       # band height [bins]
        band_start_raw, data_raw = P.band_tables_raw()
        # binned band start (round half-up); the band then covers raw rows
        # [start*b, start*b + H*b), which may sit up to b/2 raw rows away from
        # the MATLAB raw band -- tracked below as a per-source z correction.
        self.band_start = np.floor(band_start_raw / b + 0.5).astype(int)
        j0 = np.ceil(data_raw[:, 0] / b).astype(int)      # fully covered bins only
        j1 = np.floor(data_raw[:, 1] / b).astype(int)
        j0 = np.maximum(j0, -self.band_start)             # clip to the binned panel
        j1 = np.minimum(j1, self.rows - self.band_start)
        self.data_j = np.stack([j0, j1], axis=1)          # band-local, half-open

        # ---- Effective z tables [mm], incl. the +z_shift calibration ----
        # Placement shift of the binned band centre vs the MATLAB raw band, in
        # raw rows (pre-flip: +row = -z, hence the minus sign):
        center_shift = (self.band_start * b + self.H * b / 2) \
                       - (band_start_raw + P.beam_width / 2)
        self.z_src = np.asarray(P.z_src_mm) + P.z_shift_mm
        self.z_det = np.asarray(P.z_det_mm) + P.z_shift_mm - center_shift * P.pixel_size

    # ------------------------------------------------------------------ #
    def summary(self):
        P = self.params
        print(f"Binned grid   : bin {self.b}  ->  panel {self.rows} x {self.cols} px, "
              f"band {self.H} x {self.cols} px @ {self.pixel_size:.2f} mm")
        print(f"Half-fan      : offset_u = {self.offset_u:.2f} mm, "
              f"SOD/ODD = {P.SOD:.0f}/{P.ODD:.0f} mm")
        with np.printoptions(precision=2, suppress=True):
            print(f"z_src (eff)   : {self.z_src}")
            print(f"z_det (eff)   : {self.z_det}")
