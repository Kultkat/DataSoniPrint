"""
processing.py — Headless data processing engine for DataSoniPrint web.

Extracts all computation from the pygame/pyo GUI code into pure functions
that take numpy arrays + parameters and return file-like outputs.

No GUI, no audio server — just numpy math → WAV bytes / STL bytes / JSON.
"""

import csv
import io
import json
import math
import os
import struct
import tempfile
import wave
import numpy as np

# Optional format libraries — gracefully degrade if missing
try:
    import h5py
except ImportError:
    h5py = None

try:
    import netCDF4
except ImportError:
    netCDF4 = None

try:
    import cfgrib
except ImportError:
    cfgrib = None

try:
    import asdf
except ImportError:
    asdf = None


# ─── Supported extensions ────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".h5", ".hdf5", ".hdf",
    ".nc", ".nc4", ".netcdf",
    ".grib", ".grib2", ".grb", ".grb2",
    ".asdf",
}


def supported_extension(filename):
    """Check if a filename has a supported extension."""
    ext = os.path.splitext(filename.lower())[1]
    return ext in SUPPORTED_EXTENSIONS


def _detect_format(filename):
    """Return format string from filename extension."""
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".csv":
        return "csv"
    if ext in (".h5", ".hdf5", ".hdf"):
        return "hdf5"
    if ext in (".nc", ".nc4", ".netcdf"):
        return "netcdf"
    if ext in (".grib", ".grib2", ".grb", ".grb2"):
        return "grib"
    if ext == ".asdf":
        return "asdf"
    raise ValueError(f"Unsupported file extension: {ext}")


# ─── CSV loading ─────────────────────────────────────────────────────────────

def load_csv(file_stream):
    """Load CSV from an in-memory file stream.

    Returns:
        headers: list of column names (numeric columns only)
        columns: dict mapping header → numpy float64 array
        row_count: int
    """
    text = file_stream.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8-sig")

    reader = csv.reader(io.StringIO(text))
    raw_headers = next(reader, None)
    if raw_headers is None:
        raise ValueError("CSV is empty")

    rows = list(reader)
    if not rows:
        raise ValueError("CSV has headers but no data rows")

    headers = []
    columns = {}
    for ci, h in enumerate(raw_headers):
        h = h.strip() or f"col_{ci}"
        vals = []
        num_count = 0
        for row in rows:
            if ci < len(row):
                cell = row[ci].strip()
                try:
                    vals.append(float(cell))
                    num_count += 1
                except ValueError:
                    vals.append(float("nan"))
            else:
                vals.append(float("nan"))
        if num_count > len(rows) * 0.5:
            headers.append(h)
            columns[h] = np.array(vals, dtype=np.float64)

    if not headers:
        raise ValueError("No numeric columns found in CSV")

    return headers, columns, len(rows)


# ─── HDF5 loading ────────────────────────────────────────────────────────────

def load_hdf5(file_bytes):
    """Load numeric 1-D datasets from an HDF5 file.

    Returns (headers, columns, row_count).
    """
    if h5py is None:
        raise ValueError("HDF5 support requires the h5py library")

    buf = io.BytesIO(file_bytes)
    headers = []
    columns = {}
    max_len = 0

    with h5py.File(buf, "r") as f:
        def _visit(name, obj):
            nonlocal max_len
            if isinstance(obj, h5py.Dataset):
                if obj.ndim == 1 and np.issubdtype(obj.dtype, np.number):
                    data = obj[()].astype(np.float64)
                    label = name.replace("/", ".")
                    headers.append(label)
                    columns[label] = data
                    max_len = max(max_len, len(data))
                elif obj.ndim == 2 and np.issubdtype(obj.dtype, np.number):
                    data = obj[()]
                    for ci in range(min(data.shape[1], 64)):
                        label = f"{name.replace('/', '.')}.col{ci}"
                        col = data[:, ci].astype(np.float64)
                        headers.append(label)
                        columns[label] = col
                        max_len = max(max_len, len(col))
        f.visititems(_visit)

    if not headers:
        raise ValueError("No numeric datasets found in HDF5 file")

    return headers, columns, max_len


# ─── NetCDF loading ──────────────────────────────────────────────────────────

def load_netcdf(file_bytes):
    """Load numeric variables from a NetCDF file.

    Returns (headers, columns, row_count).
    """
    if netCDF4 is None:
        raise ValueError("NetCDF support requires the netCDF4 library")

    # netCDF4 needs a real file path — write to temp file
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        ds = netCDF4.Dataset(tmp_path, "r")
        headers = []
        columns = {}
        max_len = 0

        for var_name in ds.variables:
            var = ds.variables[var_name]
            if np.issubdtype(var.dtype, np.number):
                data = var[:].flatten().astype(np.float64)
                if hasattr(data, "filled"):
                    data = data.filled(np.nan)
                headers.append(var_name)
                columns[var_name] = data
                max_len = max(max_len, len(data))

        ds.close()
    finally:
        os.unlink(tmp_path)

    if not headers:
        raise ValueError("No numeric variables found in NetCDF file")

    return headers, columns, max_len


# ─── GRIB loading ────────────────────────────────────────────────────────────

def load_grib(file_bytes):
    """Load numeric fields from a GRIB/GRIB2 file.

    Returns (headers, columns, row_count).
    """
    if cfgrib is None:
        raise ValueError("GRIB support requires the cfgrib library")

    import xarray as xr

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        datasets = cfgrib.open_datasets(tmp_path)
        headers = []
        columns = {}
        max_len = 0

        for ds in datasets:
            for var_name in ds.data_vars:
                var = ds[var_name]
                if np.issubdtype(var.dtype, np.number):
                    data = var.values.flatten().astype(np.float64)
                    data = np.where(np.isfinite(data), data, np.nan)
                    headers.append(var_name)
                    columns[var_name] = data
                    max_len = max(max_len, len(data))

        for ds in datasets:
            ds.close()
    finally:
        os.unlink(tmp_path)

    if not headers:
        raise ValueError("No numeric fields found in GRIB file")

    return headers, columns, max_len


# ─── ASDF loading ────────────────────────────────────────────────────────────

def load_asdf(file_bytes):
    """Load numeric arrays from an ASDF file.

    Returns (headers, columns, row_count).
    """
    if asdf is None:
        raise ValueError("ASDF support requires the asdf library")

    with tempfile.NamedTemporaryFile(suffix=".asdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        af = asdf.open(tmp_path)
        headers = []
        columns = {}
        max_len = 0

        def _walk(tree, prefix=""):
            nonlocal max_len
            if isinstance(tree, dict):
                for key, val in tree.items():
                    _walk(val, f"{prefix}{key}." if prefix else f"{key}.")
            elif hasattr(tree, 'dtype') and hasattr(tree, 'shape') and np.issubdtype(tree.dtype, np.number):
                data = np.asarray(tree).flatten().astype(np.float64)
                label = prefix.rstrip(".")
                headers.append(label)
                columns[label] = data
                max_len = max(max_len, len(data))
            elif isinstance(tree, (list, tuple)):
                for i, item in enumerate(tree):
                    _walk(item, f"{prefix}[{i}].")

        _walk(af.tree)
        af.close()
    finally:
        os.unlink(tmp_path)

    if not headers:
        raise ValueError("No numeric arrays found in ASDF file")

    return headers, columns, max_len


# ─── Unified loader ─────────────────────────────────────────────────────────

def load_file(file_bytes, filename):
    """Detect format from filename and load data.

    Returns (headers, columns, row_count).
    """
    fmt = _detect_format(filename)

    if fmt == "csv":
        return load_csv(io.BytesIO(file_bytes))
    elif fmt == "hdf5":
        return load_hdf5(file_bytes)
    elif fmt == "netcdf":
        return load_netcdf(file_bytes)
    elif fmt == "grib":
        return load_grib(file_bytes)
    elif fmt == "asdf":
        return load_asdf(file_bytes)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


# ─── Quick preview chart ────────────────────────────────────────────────────

def generate_preview_png(columns, headers, selected_column=None):
    """Generate a quick matplotlib line chart PNG for the selected column.

    If selected_column is None, plots the first column.
    Returns PNG bytes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col_name = selected_column if selected_column in columns else headers[0]
    data = columns[col_name]

    fig, ax = plt.subplots(figsize=(8, 3), dpi=100)
    fig.patch.set_facecolor("#12121c")
    ax.set_facecolor("#0a0a12")

    ax.plot(data, color="#50b4ff", linewidth=0.6, alpha=0.9)
    ax.set_title(col_name, color="#dcdce6", fontsize=11, pad=8)
    ax.set_xlabel("Sample", color="#8c8ca0", fontsize=9)
    ax.set_ylabel("Value", color="#8c8ca0", fontsize=9)
    ax.tick_params(colors="#8c8ca0", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#32324a")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─── Data safety ─────────────────────────────────────────────────────────────

# Cap input points to prevent multi-GB memory allocations.
# 500K points × 441 samples/point = 220M samples ≈ 1.7 GB — manageable.
MAX_DATA_POINTS = 500_000

# Time per data point (seconds).  Matches csv_viewer.py: 0.01s = 10ms each.
SECONDS_PER_POINT = 0.01


def _safe_downsample(data, max_points=MAX_DATA_POINTS):
    """Downsample a 1D array to max_points via linear interpolation."""
    if len(data) <= max_points:
        return data
    old_t = np.linspace(0, 1, len(data))
    new_t = np.linspace(0, 1, max_points)
    return np.interp(new_t, old_t, data)


# ─── Column stats ────────────────────────────────────────────────────────────

def column_stats(columns, headers):
    """Return per-column min/max/mean/count."""
    stats = {}
    for h in headers:
        col = columns[h]
        valid = col[np.isfinite(col)]
        if len(valid) > 0:
            stats[h] = {
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
                "mean": float(np.mean(valid)),
                "count": int(len(valid)),
            }
    return stats


# ─── FM Synthesis ────────────────────────────────────────────────────────────

def sonify_column(data, spread=0.35, playback_sr=44100):
    """Convert a numeric column to FM-synthesized audio.

    Every data point gets SECONDS_PER_POINT (10ms) of audio — all points
    are audible.  Duration scales with the dataset:
        audio_duration = n_points × 0.01s

    This matches the LIGO sonifier / CSV-viewer approach.
    """
    data = _safe_downsample(np.nan_to_num(data.copy(), nan=0.0))

    peak = np.max(np.abs(data))
    normed = data / peak if peak > 0 else data

    # Each data point → 10ms of audio (441 samples at 44100 Hz)
    samples_per_point = max(1, int(playback_sr * SECONDS_PER_POINT))
    out_len = len(normed) * samples_per_point

    orig_t = np.linspace(0, 1, len(normed))
    new_t = np.linspace(0, 1, out_len)
    normed_up = np.interp(new_t, orig_t, normed)

    # FM synthesis: value → instantaneous frequency
    semi_range = 2 + spread * 46  # 2–48 semitones
    center_hz = 440.0
    freq_array = center_hz * np.power(2.0, normed_up * semi_range / 12.0)
    phase_inc = freq_array / playback_sr
    phase = np.cumsum(phase_inc)
    audio = np.sin(2.0 * np.pi * phase) * 0.75

    return audio


def sonify_columns(columns, col_names, spread=0.35, playback_sr=44100):
    """Sonify multiple columns as layered FM voices, mixed to mono.

    Each column is independently FM-synthesized (same approach as
    sonify_column), then all voices are summed and normalized.
    Duration is determined by the data — every point gets 10ms.
    """
    if not col_names:
        raise ValueError("No columns selected for sonification")

    n_voices = len(col_names)
    audios = []

    for name in col_names:
        voice = sonify_column(columns[name], spread=spread, playback_sr=playback_sr)
        audios.append(voice)

    # Align to shortest voice length
    min_len = min(len(a) for a in audios)
    audios = [a[:min_len] for a in audios]

    # Mix: sum all voices, normalize
    mix = np.zeros(min_len)
    for a in audios:
        mix += a

    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = mix * (0.75 / peak)

    return mix


def apply_speed(audio, speed, playback_sr=44100):
    """Resample audio to change playback speed without changing pitch mapping.

    speed: multiplier (e.g. 2.0 = twice as fast = half as many samples)
    """
    if speed == 1.0:
        return audio
    new_len = max(1, int(len(audio) / speed))
    old_t = np.linspace(0, 1, len(audio))
    new_t = np.linspace(0, 1, new_len)
    return np.interp(new_t, old_t, audio)


def apply_volume(audio, volume):
    """Scale amplitude."""
    return audio * volume


# ─── WAV export ──────────────────────────────────────────────────────────────

def audio_to_wav_bytes(audio, sample_rate=44100):
    """Convert float64 mono audio array to in-memory WAV bytes."""
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    buf.seek(0)
    return buf.read()


# ─── Spectrogram ─────────────────────────────────────────────────────────────

def compute_spectrogram(audio, sr=44100, nperseg=1024, overlap_frac=0.75):
    """Compute STFT spectrogram.

    Returns:
        times: 1D array of frame center times (seconds)
        freqs: 1D array of frequency bins (Hz)
        spec_db: 2D array (n_freqs × n_frames), power in dB
    """
    step = int(nperseg * (1 - overlap_frac))
    n_frames = (len(audio) - nperseg) // step + 1

    window = np.hanning(nperseg)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sr)

    spec = np.zeros((len(freqs), n_frames))
    times = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * step
        chunk = audio[start:start + nperseg] * window
        spec[:, i] = np.abs(np.fft.rfft(chunk))
        times[i] = start / sr

    spec_db = 20 * np.log10(spec + 1e-30)
    return times, freqs, spec_db


# ─── STL export ──────────────────────────────────────────────────────────────

def spectrogram_to_stl_bytes(times, freqs, spec_db,
                              freq_max=2000, time_downsample=4,
                              freq_downsample=2, base_thickness=2.0,
                              height_scale=15.0, width_mm=150, depth_mm=80):
    """Generate 3D-printable STL mesh from spectrogram, return as bytes.

    The spectrogram becomes a terrain surface:
    - X = time, Y = frequency, Z = power (height)
    - Solid base + side walls for printability
    """
    from stl import mesh as stl_mesh

    freq_mask = freqs <= freq_max
    spec = spec_db[freq_mask, :]
    spec = spec[::freq_downsample, ::time_downsample]
    n_freq, n_time = spec.shape

    # normalize height to [0, 1]
    vmin = np.percentile(spec, 2)
    vmax = np.percentile(spec, 99.5)
    height = np.clip((spec - vmin) / (vmax - vmin + 1e-10), 0, 1)

    x_scale = width_mm / max(n_time - 1, 1)
    y_scale = depth_mm / max(n_freq - 1, 1)

    # top surface vertices
    top_verts = np.zeros((n_freq, n_time, 3))
    for iy in range(n_freq):
        for ix in range(n_time):
            top_verts[iy, ix] = [
                ix * x_scale, iy * y_scale,
                base_thickness + height[iy, ix] * height_scale
            ]

    # bottom surface (flat)
    bot_verts = np.zeros((n_freq, n_time, 3))
    for iy in range(n_freq):
        for ix in range(n_time):
            bot_verts[iy, ix] = [ix * x_scale, iy * y_scale, 0.0]

    triangles = []

    def add_quad(v0, v1, v2, v3):
        triangles.append([v0, v1, v2])
        triangles.append([v0, v2, v3])

    # top surface
    for iy in range(n_freq - 1):
        for ix in range(n_time - 1):
            add_quad(top_verts[iy, ix], top_verts[iy, ix + 1],
                     top_verts[iy + 1, ix + 1], top_verts[iy + 1, ix])

    # bottom surface (reverse winding)
    for iy in range(n_freq - 1):
        for ix in range(n_time - 1):
            add_quad(bot_verts[iy, ix], bot_verts[iy + 1, ix],
                     bot_verts[iy + 1, ix + 1], bot_verts[iy, ix + 1])

    # side walls
    for ix in range(n_time - 1):
        add_quad(bot_verts[0, ix], bot_verts[0, ix + 1],
                 top_verts[0, ix + 1], top_verts[0, ix])
    for ix in range(n_time - 1):
        add_quad(bot_verts[-1, ix], top_verts[-1, ix],
                 top_verts[-1, ix + 1], bot_verts[-1, ix + 1])
    for iy in range(n_freq - 1):
        add_quad(bot_verts[iy, 0], top_verts[iy, 0],
                 top_verts[iy + 1, 0], bot_verts[iy + 1, 0])
    for iy in range(n_freq - 1):
        add_quad(bot_verts[iy, -1], bot_verts[iy + 1, -1],
                 top_verts[iy + 1, -1], top_verts[iy, -1])

    tri_array = np.array(triangles)
    m = stl_mesh.Mesh(np.zeros(len(tri_array), dtype=stl_mesh.Mesh.dtype))
    for i, tri in enumerate(tri_array):
        m.vectors[i] = tri

    buf = io.BytesIO()
    m.save("output.stl", fh=buf)
    buf.seek(0)
    return buf.read()


# ─── Relief STL (direct data → 3D print) ─────────────────────────────────────

def image_to_relief_stl(
    image_bytes: bytes,
    width_mm: float = 150.0,
    depth_mm: float = 150.0,
    height_scale_mm: float = 50.0,
    base_thickness_mm: float = 3.0,
    max_resolution_mm: float = 0.4,
    max_mp: int = 10,
) -> bytes:
    """Convert image (grayscale) to 3D-printable relief STL.

    Brightness → height mapping:
      - Black (0) → base_thickness_mm
      - White (255) → base_thickness_mm + height_scale_mm

    Args:
        image_bytes:        PNG/JPG bytes
        width_mm:           X extent in mm
        depth_mm:           Y extent in mm
        height_scale_mm:    max Z range above base (default 50mm = 5cm)
        base_thickness_mm:  flat base height (default 3mm)
        max_resolution_mm:  target pixel size in mm (default 0.4mm)
        max_mp:             max megapixels (default 10)

    Returns:
        Binary STL bytes.

    Raises:
        ValueError: if image is too large or invalid format.
    """
    from PIL import Image
    from stl import mesh as stl_mesh

    # --- 1. Load and validate image ---
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")

    w, h = img.size
    mp = (w * h) / 1_000_000
    if mp > max_mp:
        raise ValueError(f"Image too large: {mp:.1f}MP (max {max_mp}MP)")

    # --- 2. Convert to grayscale ---
    if img.mode != 'L':
        img = img.convert('L')

    # --- 3. Downsample to printable resolution ---
    # Target: ~max_resolution_mm per pixel
    # But keep manageable size for STL (e.g., max ~250×250 pixels)
    target_px_w = min(int(width_mm / max_resolution_mm), 250)
    target_px_h = min(int(depth_mm / max_resolution_mm), 250)
    aspect_ratio = w / h
    if aspect_ratio > (width_mm / depth_mm):
        target_px_h = int(target_px_w / aspect_ratio)
    else:
        target_px_w = int(target_px_h * aspect_ratio)

    if target_px_w < 10 or target_px_h < 10:
        target_px_w, target_px_h = 10, 10

    img = img.resize((target_px_w, target_px_h), Image.Resampling.LANCZOS)
    px_array = np.array(img, dtype=np.float64) / 255.0  # normalize [0, 1]

    n_x, n_y = px_array.shape[1], px_array.shape[0]

    # --- 4. Build vertex grid ---
    # X: 0 to width_mm, Y: 0 to depth_mm, Z: height from pixel brightness
    x_scale = width_mm / (n_x - 1) if n_x > 1 else 1.0
    y_scale = depth_mm / (n_y - 1) if n_y > 1 else 1.0

    vertices = np.zeros((n_y, n_x, 3), dtype=np.float64)
    for iy in range(n_y):
        for ix in range(n_x):
            brightness = px_array[iy, ix]
            vertices[iy, ix] = [
                ix * x_scale,
                iy * y_scale,
                float(base_thickness_mm) + brightness * float(height_scale_mm),
            ]

    # --- 5. Triangulate grid into mesh ---
    tris = []
    for iy in range(n_y - 1):
        for ix in range(n_x - 1):
            v0 = vertices[iy, ix]
            v1 = vertices[iy, ix + 1]
            v2 = vertices[iy + 1, ix + 1]
            v3 = vertices[iy + 1, ix]

            # Two triangles per quad: CCW winding (outward normal)
            tris.append([v0, v1, v2])
            tris.append([v0, v2, v3])

    # Bottom surface (flat base at z=0, reversed winding)
    for iy in range(n_y - 1):
        for ix in range(n_x - 1):
            v0 = np.array([vertices[iy, ix, 0], vertices[iy, ix, 1], 0.0])
            v1 = np.array([vertices[iy, ix + 1, 0], vertices[iy, ix + 1, 1], 0.0])
            v2 = np.array([vertices[iy + 1, ix + 1, 0], vertices[iy + 1, ix + 1, 1], 0.0])
            v3 = np.array([vertices[iy + 1, ix, 0], vertices[iy + 1, ix, 1], 0.0])

            tris.append([v0, v3, v2])
            tris.append([v0, v2, v1])

    # Edge walls (simplified: just the perimeter quads)
    # Left edge (x=0)
    for iy in range(n_y - 1):
        v0 = np.array([0.0, vertices[iy, 0, 1], 0.0])
        v1 = np.array([0.0, vertices[iy, 0, 1], vertices[iy, 0, 2]])
        v2 = np.array([0.0, vertices[iy + 1, 0, 1], vertices[iy + 1, 0, 2]])
        v3 = np.array([0.0, vertices[iy + 1, 0, 1], 0.0])
        tris.append([v0, v1, v2])
        tris.append([v0, v2, v3])

    # Right edge (x=width_mm)
    for iy in range(n_y - 1):
        v0 = np.array([width_mm, vertices[iy, -1, 1], 0.0])
        v1 = np.array([width_mm, vertices[iy, -1, 1], vertices[iy, -1, 2]])
        v2 = np.array([width_mm, vertices[iy + 1, -1, 1], vertices[iy + 1, -1, 2]])
        v3 = np.array([width_mm, vertices[iy + 1, -1, 1], 0.0])
        tris.append([v0, v3, v2])
        tris.append([v0, v2, v1])

    # Front edge (y=0)
    for ix in range(n_x - 1):
        v0 = np.array([vertices[0, ix, 0], 0.0, 0.0])
        v1 = np.array([vertices[0, ix, 0], 0.0, vertices[0, ix, 2]])
        v2 = np.array([vertices[0, ix + 1, 0], 0.0, vertices[0, ix + 1, 2]])
        v3 = np.array([vertices[0, ix + 1, 0], 0.0, 0.0])
        tris.append([v0, v1, v2])
        tris.append([v0, v2, v3])

    # Back edge (y=depth_mm)
    for ix in range(n_x - 1):
        v0 = np.array([vertices[-1, ix, 0], depth_mm, 0.0])
        v1 = np.array([vertices[-1, ix, 0], depth_mm, vertices[-1, ix, 2]])
        v2 = np.array([vertices[-1, ix + 1, 0], depth_mm, vertices[-1, ix + 1, 2]])
        v3 = np.array([vertices[-1, ix + 1, 0], depth_mm, 0.0])
        tris.append([v0, v3, v2])
        tris.append([v0, v2, v1])

    tri_array = np.array(tris, dtype=np.float64)
    n_tris = len(tri_array)

    # --- 6. Build mesh and export ---
    m = stl_mesh.Mesh(np.zeros(n_tris, dtype=stl_mesh.Mesh.dtype))
    m.vectors = tri_array
    m.update_normals()

    buf = io.BytesIO()
    m.save("relief.stl", fh=buf)
    buf.seek(0)
    return buf.read()


def scale_to_bed(width_mm: float, depth_mm: float,
                 height_scale_mm: float, max_bed_mm: float = 200.0
                 ) -> tuple:
    """Proportionally scale width and depth to fit within max_bed_mm × max_bed_mm.

    Height is never scaled — it is an independent aesthetic parameter.
    Returns (scaled_width_mm, scaled_depth_mm, scale_factor) where
    scale_factor is in (0, 1].  If both dimensions already fit, returns
    the originals unchanged with scale_factor == 1.0.

    Args:
        width_mm:       requested X dimension in mm
        depth_mm:       requested Y dimension in mm
        height_scale_mm: Z range in mm (returned unchanged, present for
                         completeness so callers can forward all 3 values)
        max_bed_mm:     maximum allowed dimension on each axis (default 200 mm)

    Returns:
        (scaled_width_mm, scaled_depth_mm, scale_factor)
    """
    scale = min(max_bed_mm / max(width_mm, 1e-9),
                max_bed_mm / max(depth_mm, 1e-9),
                1.0)
    return float(width_mm * scale), float(depth_mm * scale), float(scale)


def data_column_to_relief_stl(
    data: np.ndarray,
    width_mm: float = 150.0,
    depth_mm: float = 20.0,
    height_scale_mm: float = 20.0,
    base_thickness_mm: float = 3.0,
    n_points: int = 200,
) -> bytes:
    """Convert a 1-D data column to a watertight 3D-printable STL relief.

    The relief is a solid extrusion:
      - X axis  = sample index, scaled to [0, width_mm]
      - Y axis  = constant depth, spanning [0, depth_mm]
      - Z axis  = base_thickness_mm + normalized_value * height_scale_mm

    The mesh is watertight (manifold) and suitable for FDM/SLA printing.
    Triangle count = 8 * (n_points - 1) + 4.

    Args:
        data:               1-D numpy array of numeric values (NaN-safe)
        width_mm:           X extent of the relief in mm
        depth_mm:           Y extrusion depth in mm
        height_scale_mm:    maximum Z range above base_thickness_mm
        base_thickness_mm:  flat base height ensuring printability
        n_points:           number of profile samples (≤ len(data))

    Returns:
        Binary STL bytes ready to send as a file download.

    Raises:
        ValueError: if data is empty after NaN removal.
    """
    from stl import mesh as stl_mesh

    # --- 1. Downsample to printable resolution ---
    data = np.nan_to_num(np.asarray(data, dtype=np.float64), nan=0.0)
    if len(data) == 0:
        raise ValueError("Data column is empty")

    n_pts = min(int(n_points), len(data))
    if len(data) > n_pts:
        t_old = np.linspace(0.0, 1.0, len(data))
        t_new = np.linspace(0.0, 1.0, n_pts)
        data = np.interp(t_new, t_old, data)

    n = len(data)   # actual sample count (may be < n_points if data was short)

    # --- 2. Normalize values to [0, 1] ---
    d_min, d_max = data.min(), data.max()
    if d_max > d_min:
        norm = (data - d_min) / (d_max - d_min)
    else:
        norm = np.zeros(n, dtype=np.float64)   # flat line → all at base

    # --- 3. Build vertex arrays ---
    xs = np.linspace(0.0, float(width_mm), n)
    zs = float(base_thickness_mm) + norm * float(height_scale_mm)

    TF = np.column_stack([xs, np.zeros(n),                 zs])           # top-front
    TB = np.column_stack([xs, np.full(n, float(depth_mm)), zs])         # top-back
    BF = np.column_stack([xs, np.zeros(n),                 np.zeros(n)])  # bot-front
    BB = np.column_stack([xs, np.full(n, float(depth_mm)), np.zeros(n)]) # bot-back

    # --- 4. Triangulate all six faces (vectorized for speed) ---
    tris = []

    # Top surface  — outward normal +Z
    for i in range(n - 1):
        tris.append([TF[i], TF[i + 1], TB[i + 1]])
        tris.append([TF[i], TB[i + 1], TB[i]])

    # Bottom surface — outward normal -Z
    for i in range(n - 1):
        tris.append([BF[i], BB[i], BB[i + 1]])
        tris.append([BF[i], BB[i + 1], BF[i + 1]])

    # Front wall (y=0) — outward normal -Y
    for i in range(n - 1):
        tris.append([BF[i], BF[i + 1], TF[i + 1]])
        tris.append([BF[i], TF[i + 1], TF[i]])

    # Back wall (y=depth_mm) — outward normal +Y
    for i in range(n - 1):
        tris.append([BB[i], TB[i], TB[i + 1]])
        tris.append([BB[i], TB[i + 1], BB[i + 1]])

    # Left cap (x=0) — outward normal -X
    tris.append([BF[0], TF[0], TB[0]])
    tris.append([BF[0], TB[0], BB[0]])

    # Right cap (x=width_mm) — outward normal +X
    tris.append([BF[-1], BB[-1], TB[-1]])
    tris.append([BF[-1], TB[-1], TF[-1]])

    tri_array = np.array(tris, dtype=np.float64)
    n_tris = len(tri_array)

    # --- 5. Build mesh from triangles ---
    m = stl_mesh.Mesh(np.zeros(n_tris, dtype=stl_mesh.Mesh.dtype))
    m.vectors = tri_array
    m.update_normals()

    buf = io.BytesIO()
    m.save("relief.stl", fh=buf)
    buf.seek(0)
    return buf.read()


# ─── Full pipeline ───────────────────────────────────────────────────────────

def process_file(file_bytes, filename, params):
    """Run the complete DataSoniPrint pipeline on any supported file.

    Duration scales with the data: each data point gets 10ms of audio.
    Speed slider then compresses/expands playback time.

    Args:
        file_bytes: raw bytes of the uploaded file
        filename: original filename (used to detect format)
        params: dict with keys:
            columns: list of str — columns to sonify (layered FM voices)
            column: str — fallback single column
            spread: float 0–1
            speed: float 0–1 slider value (maps to 0.25x–4x)
            volume: float 0–1

    Returns:
        dict with wav, stl, settings, stats, headers
    """
    SR = 44100

    # --- load ---
    headers, columns, row_count = load_file(file_bytes, filename)

    # --- determine columns ---
    selected_cols = params.get("columns", [])
    single_col = params.get("column")

    if not selected_cols:
        if single_col and single_col in columns:
            selected_cols = [single_col]
        else:
            selected_cols = headers[:8]

    selected_cols = [c for c in selected_cols if c in columns]
    if not selected_cols:
        selected_cols = [headers[0]]

    spread = float(params.get("spread", 0.35))
    speed_slider = float(params.get("speed", 0.5))
    volume = float(params.get("volume", 0.7))

    speed_multiplier = 0.25 * (16 ** speed_slider)  # 0.25x–4x

    # --- sonify (all data points audible, duration = n_points × 10ms) ---
    if len(selected_cols) == 1:
        audio = sonify_column(columns[selected_cols[0]], spread=spread, playback_sr=SR)
    else:
        audio = sonify_columns(columns, selected_cols, spread=spread, playback_sr=SR)

    # Speed resamples the audio (doesn't drop data points)
    audio = apply_speed(audio, speed_multiplier, playback_sr=SR)
    audio = apply_volume(audio, volume)

    # --- WAV ---
    wav_bytes = audio_to_wav_bytes(audio, sample_rate=SR)

    # --- spectrogram + STL ---
    times, freqs, spec_db = compute_spectrogram(audio, sr=SR)
    stl_bytes = spectrogram_to_stl_bytes(times, freqs, spec_db)

    # --- settings ---
    stats = column_stats(columns, headers)
    fmt = _detect_format(filename)
    settings = {
        "source_file": filename,
        "source_format": fmt,
        "row_count": row_count,
        "columns_sonified": selected_cols,
        "n_voices": len(selected_cols),
        "spread": spread,
        "speed_slider": speed_slider,
        "speed_multiplier": round(speed_multiplier, 4),
        "volume": volume,
        "sample_rate": SR,
        "audio_samples": len(audio),
        "audio_duration_sec": round(len(audio) / SR, 2),
        "seconds_per_point": SECONDS_PER_POINT,
        "stl_dimensions_mm": "150 x 80 x 17",
        "column_stats": stats,
    }

    return {
        "wav": wav_bytes,
        "stl": stl_bytes,
        "settings": settings,
        "stats": stats,
        "headers": headers,
    }
