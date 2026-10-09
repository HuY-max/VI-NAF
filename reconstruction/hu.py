"""hu.py — mu [cm^-1] -> Hounsfield units, and the HU volume as a CT DICOM series.

    HU = (mu - mu_water) / (mu_water - mu_air) * 1000          (all in cm^-1)

``mu_water`` / ``mu_air`` are the linear attenuation coefficients at the mean
energy of the beam, read from ``attenuation_water_air.csv`` (0.5 keV steps,
linear interpolation in between).  The mean energy is a property of the scan,
set in each notebook's parameter cell: 95 keV for CBCT, 67 keV for msCBCT.

DICOM: one file per axial slice, CT Image Storage, Explicit VR Little Endian,
signed int16 with RescaleSlope 1 / RescaleIntercept 0, i.e. stored value = HU.
A (z, y, x) volume maps to

    slice k  ->  pixel array  vol[k]      (rows = y, columns = x)

with ``ImageOrientationPatient = [1,0,0, 0,1,0]`` and the volume centred on the
rotation axis: the first voxel of slice k sits at
``(-(nx-1)/2*dx, -(ny-1)/2*dy, -(nz-1)/2*dz + k*dz)`` [mm].  The energy and
mu_water / mu_air used go into ``ImageComments``, so the series documents its
own conversion.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, PYDICOM_IMPLEMENTATION_UID, generate_uid

ATTENUATION_TABLE = Path(__file__).resolve().parent / "attenuation_water_air.csv"
CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"
INT16_MIN, INT16_MAX = -32768, 32767


def mu_water_air(energy_kev):
    """(mu_water, mu_air) [cm^-1] at ``energy_kev``."""
    e, water, air = np.loadtxt(ATTENUATION_TABLE, delimiter=",", unpack=True)
    if not e[0] <= energy_kev <= e[-1]:
        raise ValueError(f"mean energy {energy_kev} keV is outside the table "
                         f"[{e[0]:g}, {e[-1]:g}] keV")
    return float(np.interp(energy_kev, e, water)), float(np.interp(energy_kev, e, air))


def mu_to_hu(mu, energy_kev):
    """mu [cm^-1] -> HU (float32) at the mean energy ``energy_kev``."""
    mw, ma = mu_water_air(energy_kev)
    print(f"HU @ {energy_kev:g} keV: mu_water = {mw:.5f}, mu_air = {ma:.6f} cm^-1")
    return ((mu - mw) / (mw - ma) * 1000.0).astype(np.float32)


def _ds(value):
    """DICOM DS values are limited to 16 characters."""
    return round(float(value), 6)


def _lo(value, limit=64):
    """DICOM LO / PN values: ASCII, no backslash, length-capped."""
    s = str(value).replace("\\", "_").replace("\n", " ").strip()
    return s.encode("ascii", "replace").decode("ascii")[:limit]


def write_dicom_series(hu, out_dir, voxel_mm, *, energy_kev, phantom, device,
                       kvp, tube_current_ma, description=""):
    """Write a (z, y, x) HU volume as a CT DICOM series, one file per slice.

    Tags carry the phantom name (PatientName / PatientID), the device
    (ManufacturerModelName / StationName), kVp and tube current [mA].
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nz, ny, nx = hu.shape
    d = float(voxel_mm)
    lo, hi = float(hu.min()), float(hu.max())
    if lo < INT16_MIN or hi > INT16_MAX:
        print(f"  ! HU range [{lo:.0f}, {hi:.0f}] exceeds int16; values are clipped")

    mw, ma = mu_water_air(energy_kev)
    phantom = _lo(phantom)
    study_uid, series_uid, frame_uid = generate_uid(), generate_uid(), generate_uid()
    now = datetime.now()
    date, time_ = now.strftime("%Y%m%d"), now.strftime("%H%M%S.%f")[:13]
    x0, y0, z0 = -(nx - 1) / 2 * d, -(ny - 1) / 2 * d, -(nz - 1) / 2 * d

    for k in range(nz):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CT_IMAGE_STORAGE
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SpecificCharacterSet = "ISO_IR 100"

        # ---- identification ----
        ds.SOPClassUID = CT_IMAGE_STORAGE
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Modality = "CT"
        ds.ImageType = ["DERIVED", "SECONDARY", "AXIAL"]
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.PositionReferenceIndicator = ""
        ds.StudyID, ds.SeriesNumber, ds.InstanceNumber = "1", 1, k + 1
        ds.AccessionNumber = ds.ReferringPhysicianName = ""
        ds.StudyDate = ds.SeriesDate = ds.ContentDate = date
        ds.StudyTime = ds.SeriesTime = ds.ContentTime = time_
        ds.StudyDescription = "VI-NAF reconstruction"
        ds.SeriesDescription = _lo(f"{phantom} HU @ {energy_kev:g} keV")
        ds.ImageComments = _lo(f"HU from mu [cm^-1] @ {energy_kev:g} keV, "
                               f"mu_water={mw:.5f}, mu_air={ma:.6f}, {description}",
                               10240)

        # ---- phantom, device, kVp, tube current ----
        ds.PatientName = phantom
        ds.PatientID = phantom
        ds.PatientBirthDate = ds.PatientSex = ""
        ds.PatientPosition = "HFS"
        ds.Manufacturer = ""
        ds.ManufacturerModelName = _lo(device)
        ds.StationName = _lo(device, 16)
        ds.KVP = _ds(kvp)
        ds.XRayTubeCurrent = int(round(float(tube_current_ma)))
        ds.XRayTubeCurrentInmA = float(tube_current_ma)

        # ---- geometry ----
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [_ds(x0), _ds(y0), _ds(z0 + k * d)]
        ds.SliceLocation = _ds(z0 + k * d)
        ds.PixelSpacing = [_ds(d), _ds(d)]
        ds.SliceThickness = ds.SpacingBetweenSlices = _ds(d)

        # ---- pixel data: stored value = HU ----
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows, ds.Columns = ny, nx
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1                      # signed
        ds.RescaleSlope, ds.RescaleIntercept, ds.RescaleType = 1, 0, "HU"
        ds.WindowCenter, ds.WindowWidth = 40, 400
        ds.PixelData = np.clip(np.rint(hu[k]), INT16_MIN, INT16_MAX) \
            .astype(np.int16).tobytes()

        ds.save_as(str(out_dir / f"IM{k + 1:04d}.dcm"), enforce_file_format=True)

    print(f"saved {out_dir}/  ({nz} slices of {ny} x {nx} @ {d:g} mm)")
    return out_dir
