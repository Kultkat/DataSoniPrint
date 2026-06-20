"""
DataSoniPrint Core Engine — Pure Python processing pipeline.

Phases:
    1. Data Ingestion & Normalization (0-1 scale)
    2. Formula Engine (sympy-based equation evaluation)
    3. Image-to-Relief Converter (grayscale → height)
    4. STL Generation (trimesh, 1mm base, 200mm bounding box)
"""

import io
import numpy as np
from pathlib import Path
import tempfile

# Optional format libraries
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

try:
    import sympy as sp
    from sympy import sympify, symbols
except ImportError:
    sp = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import trimesh
except ImportError:
    trimesh = None


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: Data Ingestion & Normalization
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(file_bytes):
    """Load CSV, return headers and column data dict."""
    import csv
    text = file_bytes.decode('utf-8', errors='ignore')
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)

    if not rows:
        raise ValueError("CSV is empty")

    headers = rows[0]
    data_rows = rows[1:]

    # Convert to columns (dict of column_name → np.array)
    columns = {}
    for col_idx, header in enumerate(headers):
        col_data = []
        for row in data_rows:
            try:
                val = float(row[col_idx]) if col_idx < len(row) else 0.0
                col_data.append(val)
            except (ValueError, IndexError):
                col_data.append(0.0)
        columns[header] = np.array(col_data, dtype=np.float64)

    return headers, columns


def load_hdf5(file_bytes):
    """Load HDF5, return headers and column data dict. Skip metadata, only keep numeric 1D/2D arrays."""
    if h5py is None:
        raise ImportError("h5py not installed")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        with h5py.File(tmp_path, 'r') as f:
            headers = []
            columns = {}

            def extract_datasets(group, prefix=''):
                try:
                    for key in group.keys():
                        try:
                            path = f"{prefix}/{key}" if prefix else key
                            item = group[key]

                            # Skip known metadata groups
                            if key.lower() in {'meta', 'metadata', 'attrs', 'attributes', 'info'}:
                                continue

                            if isinstance(item, h5py.Dataset):
                                # Only process numeric, 1D or 2D datasets
                                if item.dtype.kind in {'f', 'i', 'u'}:  # float, int, uint
                                    try:
                                        data = np.array(item[:], dtype=np.float64)
                                        if data.size > 0:  # Not empty
                                            headers.append(path)
                                            columns[path] = data.flatten()
                                    except (ValueError, TypeError, MemoryError):
                                        # Skip datasets that can't be converted
                                        pass

                            elif isinstance(item, h5py.Group):
                                extract_datasets(item, path)
                        except Exception:
                            # Skip individual items that cause errors
                            pass
                except Exception:
                    pass

            extract_datasets(f)

            if not headers:
                raise ValueError("No numeric datasets found in HDF5 file. Try uploading a CSV instead.")

            return headers, columns
    finally:
        Path(tmp_path).unlink()


def load_netcdf(file_bytes):
    """Load NetCDF, return headers and column data dict."""
    if netCDF4 is None:
        raise ImportError("netCDF4 not installed")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        ds = netCDF4.Dataset(tmp_path, 'r')
        headers = list(ds.variables.keys())
        columns = {}

        for var_name in headers:
            try:
                data = ds.variables[var_name][:]
                columns[var_name] = np.array(data, dtype=np.float64).flatten()
            except:
                pass

        ds.close()

        if not columns:
            raise ValueError("No variables found in NetCDF")

        return headers, columns
    finally:
        Path(tmp_path).unlink()


def load_grib(file_bytes):
    """Load GRIB, return headers and column data dict."""
    if cfgrib is None:
        raise ImportError("cfgrib not installed")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.grib2') as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        import xarray as xr
        ds = xr.open_dataset(tmp_path, engine='cfgrib')
        headers = list(ds.data_vars.keys())
        columns = {}

        for var_name in headers:
            try:
                data = ds[var_name].values
                columns[var_name] = np.array(data, dtype=np.float64).flatten()
            except:
                pass

        ds.close()

        if not columns:
            raise ValueError("No variables found in GRIB")

        return headers, columns
    finally:
        Path(tmp_path).unlink()


def load_asdf(file_bytes):
    """Load ASDF, return headers and column data dict."""
    if asdf is None:
        raise ImportError("asdf not installed")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.asdf') as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        with asdf.open(tmp_path) as af:
            headers = []
            columns = {}

            def extract_arrays(obj, prefix=''):
                if isinstance(obj, dict):
                    for key, val in obj.items():
                        path = f"{prefix}.{key}" if prefix else key
                        if isinstance(val, np.ndarray):
                            headers.append(path)
                            columns[path] = np.array(val, dtype=np.float64).flatten()
                        elif isinstance(val, (dict, list)):
                            extract_arrays(val, path)

            extract_arrays(af.tree)

            if not headers:
                raise ValueError("No arrays found in ASDF")

            return headers, columns
    finally:
        Path(tmp_path).unlink()


def load_file(file_bytes, filename):
    """Detect format and load file. Return headers, columns dict."""
    ext = Path(filename).suffix.lower()

    if ext == '.csv':
        return load_csv(file_bytes)
    elif ext in {'.h5', '.hdf5', '.hdf'}:
        return load_hdf5(file_bytes)
    elif ext in {'.nc', '.nc4', '.netcdf'}:
        return load_netcdf(file_bytes)
    elif ext in {'.grib', '.grib2', '.grb', '.grb2'}:
        return load_grib(file_bytes)
    elif ext == '.asdf':
        return load_asdf(file_bytes)
    else:
        raise ValueError(f"Unsupported format: {ext}")


def normalize_to_01(data):
    """Normalize array to [0.0, 1.0] range. NaN → 0."""
    data = np.nan_to_num(np.asarray(data, dtype=np.float64), nan=0.0)

    d_min, d_max = data.min(), data.max()
    if d_max <= d_min:
        return np.zeros_like(data)

    return (data - d_min) / (d_max - d_min)


def select_columns_for_axes(columns, headers, x_col=None, y_col=None, z_col=None):
    """
    User selects which column maps to X, Y, Z.
    Returns normalized 1D or 2D arrays.
    """
    available = [h for h in headers if h in columns]

    if not available:
        raise ValueError("No valid columns found")

    # Fallback to first column if not specified
    x_col = x_col or available[0]
    y_col = y_col or (available[1] if len(available) > 1 else available[0])
    z_col = z_col or (available[2] if len(available) > 2 else available[0])

    x_data = normalize_to_01(columns[x_col])
    y_data = normalize_to_01(columns[y_col])
    z_data = normalize_to_01(columns[z_col])

    return x_data, y_data, z_data


def spread_to_grid(x_data, y_data, z_data, resolution=100, mode="reshape"):
    """
    Spread 1D column data into a 2D relief grid (normalized to [0,1]).

    Modes:
        "reshape"   — resample Z to resolution*resolution samples and reshape
                      (works for time-series and any 1D signal).
        "scatter"   — treat (X, Y, Z) as scattered points and interpolate onto
                      a regular grid. Requires scipy; falls back to reshape if
                      scipy is missing or X/Y are degenerate.

    Returns a (resolution, resolution) float array.
    """
    z_data = np.asarray(z_data, dtype=np.float64).flatten()
    if z_data.size == 0:
        raise ValueError("Z column is empty")

    if mode == "scatter":
        try:
            from scipy.interpolate import griddata
            x_data = np.asarray(x_data, dtype=np.float64).flatten()
            y_data = np.asarray(y_data, dtype=np.float64).flatten()
            n = min(x_data.size, y_data.size, z_data.size)
            x_data, y_data, z_data = x_data[:n], y_data[:n], z_data[:n]

            # Need X and Y to span 2D space (not colinear) for Delaunay
            x_span = float(np.ptp(x_data))
            y_span = float(np.ptp(y_data))
            colinear = False
            if x_span > 1e-9 and y_span > 1e-9:
                # Pearson correlation: |r| ≈ 1 means colinear, scatter degenerate
                xc = x_data - x_data.mean()
                yc = y_data - y_data.mean()
                denom = np.sqrt((xc * xc).sum() * (yc * yc).sum())
                if denom > 0:
                    r = float((xc * yc).sum() / denom)
                    colinear = abs(r) > 0.999

            if x_span > 1e-9 and y_span > 1e-9 and not colinear:
                xi = np.linspace(x_data.min(), x_data.max(), resolution)
                yi = np.linspace(y_data.min(), y_data.max(), resolution)
                xx, yy = np.meshgrid(xi, yi)
                try:
                    grid = griddata(
                        (x_data, y_data), z_data, (xx, yy),
                        method="linear", fill_value=np.nan,
                    )
                    if np.isnan(grid).any():
                        nearest = griddata(
                            (x_data, y_data), z_data, (xx, yy), method="nearest"
                        )
                        grid = np.where(np.isnan(grid), nearest, grid)
                    return normalize_to_01(grid)
                except Exception:
                    pass  # Delaunay failure → fall through to reshape
        except ImportError:
            pass  # scipy missing → fall through to reshape

    # Reshape mode: project 1D z onto a resolution × resolution grid.
    target_len = resolution * resolution
    if z_data.size == target_len:
        resampled = z_data
    elif z_data.size >= target_len * 4:
        # Source is much larger than target — point-sampling would alias and
        # create patchy "half flat / half spiky" textures on noise-like data.
        # Take the RMS of each window so each grid cell reflects the local
        # signal energy, giving a visually balanced relief.
        edges = np.linspace(0, z_data.size, target_len + 1, dtype=np.int64)
        resampled = np.empty(target_len, dtype=np.float64)
        for k in range(target_len):
            chunk = z_data[edges[k]:edges[k + 1]]
            if chunk.size:
                resampled[k] = np.sqrt(np.mean(chunk * chunk))
            else:
                resampled[k] = 0.0
    else:
        # Modest size mismatch — linear interpolation is fine.
        src_idx = np.linspace(0, 1, z_data.size)
        dst_idx = np.linspace(0, 1, target_len)
        resampled = np.interp(dst_idx, src_idx, z_data)

    grid = resampled.reshape(resolution, resolution)
    return normalize_to_01(grid)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: Formula Engine
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_formula(formula_str, x_range=(-5, 5), y_range=(-5, 5), resolution=50):
    """
    Evaluate a sympy formula like 'sin(x)*cos(y)' on a grid.
    Returns normalized 2D array (z values).
    """
    if sp is None:
        raise ImportError("sympy not installed")

    x, y = symbols('x y', real=True)

    try:
        # Bind x/y so the parsed expression uses OUR symbols (not fresh ones).
        expr = sympify(formula_str, locals={"x": x, "y": y})
    except Exception as e:
        raise ValueError(f"Invalid formula: {e}")

    # Generate grid
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    xx, yy = np.meshgrid(x_vals, y_vals)

    # Vectorized evaluation over the whole grid via numpy.
    try:
        f = sp.lambdify((x, y), expr, modules=["numpy"])
        zz = np.asarray(f(xx, yy), dtype=float)
        if zz.shape != xx.shape:  # formula independent of x and/or y → scalar
            zz = np.broadcast_to(zz, xx.shape).astype(float)
    except Exception as e:
        raise ValueError(f"Could not evaluate formula: {e}")

    # Drop non-finite values (e.g. log of negatives) so they don't skew scaling.
    zz = np.where(np.isfinite(zz), zz, np.nan)

    # Normalize to [0, 1]
    return normalize_to_01(zz)


def solidify_surface(z_grid, base_height_mm=1.0):
    """
    Take a zero-thickness surface (2D array of heights),
    extrude downward to create a solid block.
    Adds base_height_mm at Z=0.

    Returns 3D coordinate arrays: (verts, faces) for trimesh.
    """
    if trimesh is None:
        raise ImportError("trimesh not installed")

    # z_grid is (rows, cols) with values in [0, 1]
    h, w = z_grid.shape

    # Map to physical coordinates
    x_coords = np.linspace(0, 100, w)  # 100mm width
    y_coords = np.linspace(0, 100, h)  # 100mm depth

    vertices = []
    faces = []

    # Top surface
    for i in range(h):
        for j in range(w):
            z_val = base_height_mm + z_grid[i, j] * 50  # 50mm max height
            vertices.append([x_coords[j], y_coords[i], z_val])

    # Bottom surface (flat base)
    for i in range(h):
        for j in range(w):
            vertices.append([x_coords[j], y_coords[i], 0.0])

    # Create faces
    top_stride = w
    bot_stride = h * w

    # Top faces
    for i in range(h - 1):
        for j in range(w - 1):
            v0 = i * w + j
            v1 = i * w + (j + 1)
            v2 = (i + 1) * w + (j + 1)
            v3 = (i + 1) * w + j

            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    # Bottom faces (reversed winding for outside-facing normals)
    for i in range(h - 1):
        for j in range(w - 1):
            v0 = bot_stride + i * w + j
            v1 = bot_stride + i * w + (j + 1)
            v2 = bot_stride + (i + 1) * w + (j + 1)
            v3 = bot_stride + (i + 1) * w + j

            faces.append([v0, v2, v1])
            faces.append([v0, v3, v2])

    # Side faces
    for j in range(w - 1):
        v0_top = j
        v1_top = j + 1
        v0_bot = bot_stride + j
        v1_bot = bot_stride + j + 1

        faces.append([v0_top, v1_top, v1_bot])
        faces.append([v0_top, v1_bot, v0_bot])

    for i in range(h - 1):
        v0_top = i * w
        v1_top = (i + 1) * w
        v0_bot = bot_stride + i * w
        v1_bot = bot_stride + (i + 1) * w

        faces.append([v0_top, v0_bot, v1_bot])
        faces.append([v0_top, v1_bot, v1_top])

    return np.array(vertices), np.array(faces)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: Image-to-Relief
# ─────────────────────────────────────────────────────────────────────────────

# Hard cap on the longest image side before it becomes a mesh. Each pixel
# becomes ~2 vertices and ~4 mesh faces in scale_to_bed, so an uncapped
# multi-megapixel photo produces tens of millions of faces and OOM-kills the
# (1 GB) Streamlit Cloud worker. 800 px keeps the worst case near ~2M faces
# while still giving ~8 px/mm of detail on a 100 mm plate (ample for braille).
MAX_RELIEF_DIM = 800


def _load_grayscale(image_path, max_dim=MAX_RELIEF_DIM):
    """
    Open an image as grayscale, compositing transparency onto a WHITE background.

    Plots exported with transparency (e.g. matplotlib ``savefig(transparent=True)``)
    often store the drawing entirely in the alpha channel with a black luminance
    channel — a naive ``convert("L")`` would then return all-black and produce a
    single flat platform. Flattening onto white first preserves the real shape.

    Images larger than ``max_dim`` on their longest side are downsampled
    (preserving aspect ratio) so the resulting mesh stays within memory. Because
    both OCR (:func:`detect_text_regions`) and the relief builder go through this
    one function, text-box pixel coordinates always match the relief grid.
    """
    if Image is None:
        raise ImportError("Pillow not installed")
    img = Image.open(image_path)
    has_alpha = img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    )
    if has_alpha:
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    img = img.convert("L")
    if max_dim and max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    return img


def image_to_relief_array(image_path, invert=True, bg_threshold=0.1):
    """
    Convert an image to a normalized relief grid in [0, 1].

    The image is read as grayscale (brightness 0-255). With ``invert=True``
    (default) dark pixels become HIGH and light pixels become LOW — so the
    data drawn on a typical white-background plot becomes a raised relief and
    the background flattens to the base.

    ``bg_threshold`` (0-1) is a background cutoff applied AFTER inversion:
    everything below the cutoff is clamped to 0 (the flat build-plate base),
    so faint background texture and anti-aliasing halos don't lift the plate.

    Returns a (H, W) float array normalized to [0, 1], where 0 == background
    (base only) and 1 == the tallest data feature.
    """
    img = _load_grayscale(image_path)  # Grayscale (B&W), alpha flattened to white
    v = np.array(img, dtype=np.float64) / 255.0  # → [0, 1]

    if invert:
        v = 1.0 - v  # dark data → high, white background → low

    # Flatten background: drop everything below the cutoff to 0.
    if bg_threshold > 0:
        v = np.clip(v - bg_threshold, 0.0, None)

    return normalize_to_01(v)


def detect_text_regions(image_path, min_confidence=40):
    """
    OCR the image and return word bounding boxes as
    ``[(x, y, w, h, text), ...]`` in image-pixel coordinates.

    Requires the Tesseract engine + the ``pytesseract`` binding. Raises a clear,
    actionable error if either is missing so the caller can fall back gracefully.
    """
    try:
        import pytesseract
    except ImportError as e:
        raise ImportError(
            "pytesseract not installed — run `pip install pytesseract`."
        ) from e
    if Image is None:
        raise ImportError("Pillow not installed")

    img = _load_grayscale(image_path)
    try:
        data = pytesseract.image_to_data(
            img, output_type=pytesseract.Output.DICT
        )
    except Exception as e:  # TesseractNotFoundError and friends
        raise RuntimeError(
            "Tesseract OCR engine not found. Install it with "
            "`sudo apt-get install tesseract-ocr` (Linux) or "
            "`brew install tesseract` (macOS)."
        ) from e

    boxes = []
    for i in range(len(data["text"])):
        txt = (data["text"][i] or "").strip()
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        if txt and conf >= min_confidence:
            boxes.append((
                int(data["left"][i]), int(data["top"][i]),
                int(data["width"][i]), int(data["height"][i]), txt,
            ))
    return boxes


def build_image_relief(image_path, invert=True, bg_threshold=0.1,
                       relief_min_mm=0.1, relief_max_mm=5.0,
                       binary=False, text_boxes=None,
                       glyph_level=0.5, box_pad=2):
    """
    Build the DATA relief grid from a plot image (labels applied separately).

    The result is normalized as a FRACTION of the model's height scale so that
    ``base_mm + grid * height_mm`` yields physical millimetres. Two styles:

      - grayscale (``binary=False``): pixel brightness becomes a VARIABLE height,
        ``relief_min_mm..relief_max_mm`` — a full 3D relief (colour/shade → Z).
      - binary (``binary=True``): every data pixel gets the SAME height (grid=1.0,
        i.e. one flat step) — a uniform B/W relief you can see and feel. The
        actual step height is set by the caller's height scale.

    Background flattens to 0 (the base plate). If ``text_boxes`` (from
    :func:`detect_text_regions`) is given, those regions are EXCLUDED from the
    data relief so labels aren't raised as data — labels are rendered later by
    :func:`apply_labels` once the physical mm dimensions are known.

    Returns ``(grid, glyph_mask)`` where ``glyph_mask`` marks the dark ink
    strokes inside the text boxes (used by the "engrave" label mode). Both are in
    image (row 0 = top) orientation; flip vertically at render time to match the
    source image.
    """
    img = _load_grayscale(image_path)
    gray = np.array(img, dtype=np.float64) / 255.0  # [0,1], 0 = black ink
    v = (1.0 - gray) if invert else gray            # data (dark) → high
    h, w = v.shape

    relief_max_mm = max(relief_max_mm, 0.1)
    relief_min_mm = min(max(relief_min_mm, 0.0), relief_max_mm)

    # Text-region mask (rectangles) + glyph mask (dark ink within them).
    text_region = np.zeros((h, w), dtype=bool)
    glyph = np.zeros((h, w), dtype=bool)
    for box in (text_boxes or []):
        x, y, bw, bh = box[0], box[1], box[2], box[3]
        x0, y0 = max(0, x - box_pad), max(0, y - box_pad)
        x1, y1 = min(w, x + bw + box_pad), min(h, y + bh + box_pad)
        if x1 <= x0 or y1 <= y0:
            continue
        text_region[y0:y1, x0:x1] = True
        glyph[y0:y1, x0:x1] = v[y0:y1, x0:x1] > glyph_level

    # Data relief everywhere EXCEPT the text regions.
    data = v.copy()
    if bg_threshold > 0:
        data = np.clip(data - bg_threshold, 0.0, None)
    data[text_region] = 0.0
    data = normalize_to_01(data)  # [0,1] across the data only

    grid = np.zeros((h, w), dtype=np.float64)
    data_mask = data > 0
    if binary:
        grid[data_mask] = 1.0  # single uniform step for every data pixel
    else:
        grid[data_mask] = (
            relief_min_mm + data[data_mask] * (relief_max_mm - relief_min_mm)
        ) / relief_max_mm

    return grid, glyph


# ─────────────────────────────────────────────────────────────────────────────
# Axis labels — engrave (text) or emboss (braille)
# ─────────────────────────────────────────────────────────────────────────────

# 6-dot braille cell, dot numbering:
#     1 4
#     2 5
#     3 6
# Each entry maps a character to the dots that are raised.
_BRAILLE_LETTERS = {
    "a": (1,), "b": (1, 2), "c": (1, 4), "d": (1, 4, 5), "e": (1, 5),
    "f": (1, 2, 4), "g": (1, 2, 4, 5), "h": (1, 2, 5), "i": (2, 4), "j": (2, 4, 5),
    "k": (1, 3), "l": (1, 2, 3), "m": (1, 3, 4), "n": (1, 3, 4, 5), "o": (1, 3, 5),
    "p": (1, 2, 3, 4), "q": (1, 2, 3, 4, 5), "r": (1, 2, 3, 5), "s": (2, 3, 4),
    "t": (2, 3, 4, 5), "u": (1, 3, 6), "v": (1, 2, 3, 6), "w": (2, 4, 5, 6),
    "x": (1, 3, 4, 6), "y": (1, 3, 4, 5, 6), "z": (1, 3, 5, 6),
}
# Digits reuse the a–j shapes, introduced by the number sign.
_BRAILLE_DIGITS = {
    "1": "a", "2": "b", "3": "c", "4": "d", "5": "e",
    "6": "f", "7": "g", "8": "h", "9": "i", "0": "j",
}
_BRAILLE_PUNCT = {
    ".": (2, 5, 6), ",": (2,), "-": (3, 6), ":": (2, 5), ";": (2, 3),
    "(": (1, 2, 6), ")": (3, 4, 5), "/": (3, 4), "%": (1, 4, 6),
}
_BRAILLE_NUMBER_SIGN = (3, 4, 5, 6)
_BRAILLE_CAPITAL_SIGN = (6,)


def text_to_braille_cells(text):
    """
    Translate a string into a list of Grade-1 (uncontracted) braille cells.

    Each cell is a tuple of raised dot numbers (1–6); ``()`` is a blank space.
    Digits are preceded by the number sign, capitals by the capital sign.
    """
    cells = []
    in_number = False
    for ch in text:
        if ch == " ":
            cells.append(())
            in_number = False
            continue
        if ch.isupper():
            cells.append(_BRAILLE_CAPITAL_SIGN)
            ch = ch.lower()
        if ch.isdigit():
            if not in_number:
                cells.append(_BRAILLE_NUMBER_SIGN)
                in_number = True
            cells.append(_BRAILLE_LETTERS[_BRAILLE_DIGITS[ch]])
            continue
        in_number = False
        if ch in _BRAILLE_LETTERS:
            cells.append(_BRAILLE_LETTERS[ch])
        elif ch in _BRAILLE_PUNCT:
            cells.append(_BRAILLE_PUNCT[ch])
        # unknown characters are silently skipped
    return cells


def _stamp_dot(grid, cx, cy, rx, ry, value):
    """Raise an elliptical dot footprint in the grid to at least ``value``."""
    h, w = grid.shape
    x0, x1 = max(0, int(cx - rx)), min(w, int(np.ceil(cx + rx)) + 1)
    y0, y1 = max(0, int(cy - ry)), min(h, int(np.ceil(cy + ry)) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.ogrid[y0:y1, x0:x1]
    rx = max(rx, 0.5)
    ry = max(ry, 0.5)
    mask = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1.0
    sub = grid[y0:y1, x0:x1]
    sub[mask] = np.maximum(sub[mask], value)
    grid[y0:y1, x0:x1] = sub


def apply_labels(grid, glyph_mask, text_boxes, mode,
                 width_mm, depth_mm, height_mm,
                 engrave_depth_mm=0.3, base_mm=1.0,
                 dot_height_mm=0.6, dot_dia_mm=1.5,
                 dot_pitch_mm=2.5, cell_pitch_mm=6.0):
    """
    Apply axis labels onto a copy of the data relief ``grid`` and return it.

    ``grid`` values are fractions of ``height_mm`` (so ``value * height_mm`` is
    millimetres above the plate top). ``mode`` is one of:

      - ``"none"``    — return the grid unchanged.
      - ``"engrave"`` — carve the glyph ink (``glyph_mask``) ``engrave_depth_mm``
                        into the plate (negative values).
      - ``"braille"`` — emboss standard-size raised braille dots for each detected
                        label's recognized text, at its location.

    Braille geometry is in absolute millimetres, converted to grid pixels via the
    physical ``width_mm``/``depth_mm`` and the grid resolution — so dots stay the
    correct tactile size regardless of image resolution.
    """
    out = grid.copy()
    height_mm = max(height_mm, 0.1)

    if mode == "engrave":
        depth = max(0.0, min(engrave_depth_mm, base_mm - 0.2))
        if depth > 0 and glyph_mask is not None:
            out[glyph_mask] = -depth / height_mm
        return out

    if mode == "braille":
        h, w = out.shape
        mmx = width_mm / max(w, 1)   # mm per pixel (x)
        mmy = depth_mm / max(h, 1)   # mm per pixel (y)
        rx = (dot_dia_mm / 2.0) / mmx
        ry = (dot_dia_mm / 2.0) / mmy
        value = dot_height_mm / height_mm
        # dot number → (column, row) within the cell
        dot_pos = {1: (0, 0), 2: (0, 1), 3: (0, 2),
                   4: (1, 0), 5: (1, 1), 6: (1, 2)}
        for box in (text_boxes or []):
            x, y, bw, bh = box[0], box[1], box[2], box[3]
            text = box[4] if len(box) > 4 else ""
            cells = text_to_braille_cells(text)
            if not cells:
                continue
            # Anchor: start at the box's left, vertically centred on the box.
            start_x_mm = x * mmx
            mid_y_mm = (y + bh / 2.0) * mmy
            top_y_mm = mid_y_mm - dot_pitch_mm  # 3-row cell centred on the box
            for ci, cell in enumerate(cells):
                cell_x_mm = start_x_mm + ci * cell_pitch_mm
                for dot in cell:
                    col, row = dot_pos[dot]
                    cx_mm = cell_x_mm + col * dot_pitch_mm
                    cy_mm = top_y_mm + row * dot_pitch_mm
                    _stamp_dot(out, cx_mm / mmx, cy_mm / mmy, rx, ry, value)
        return out

    return out  # "none" or unknown


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: STL Generation & Bounding Box
# ─────────────────────────────────────────────────────────────────────────────

def scale_to_bed(z_grid, target_width_mm=100, target_depth_mm=100,
                  target_height_mm=30, max_bed_mm=200.0):
    """
    Scale a normalized 2D z_grid (values 0-1) to fit within 200×200×200mm bed.

    Returns: (scaled_mesh, actual_dims)
    """
    if trimesh is None:
        raise ImportError("trimesh not installed")

    h, w = z_grid.shape

    # Calculate scaling factor to fit in bed
    scale = min(
        max_bed_mm / max(target_width_mm, 1),
        max_bed_mm / max(target_depth_mm, 1),
        1.0
    )

    scaled_w = target_width_mm * scale
    scaled_d = target_depth_mm * scale
    scaled_h = target_height_mm * scale

    # Build mesh. This is fully vectorized with NumPy: a pure-Python double loop
    # over every pixel takes minutes for an 800px relief (~2M faces) and makes
    # the app look hung. The vertex layout and face winding below are identical
    # to the original loop version, so the resulting watertight mesh is unchanged.
    x_coords = np.linspace(0, scaled_w, w)
    y_coords = np.linspace(0, scaled_d, h)
    xx, yy = np.meshgrid(x_coords, y_coords)  # both (h, w), row-major like the loop

    # Vertices: top surface (1mm base + scaled height) then bottom surface (z=0).
    top = np.stack([xx, yy, 1.0 + z_grid * scaled_h], axis=-1).reshape(-1, 3)
    bot = np.stack([xx, yy, np.zeros_like(xx)], axis=-1).reshape(-1, 3)
    vertices = np.vstack([top, bot])

    bot_start = h * w
    idx = np.arange(bot_start).reshape(h, w)  # vertex index of each top grid cell

    # Top faces (two triangles per quad, outward normal +Z).
    a = idx[:-1, :-1].ravel()
    b = idx[:-1, 1:].ravel()
    c = idx[1:, 1:].ravel()
    d = idx[1:, :-1].ravel()
    top_faces = np.vstack([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])

    # Bottom faces (reversed winding so the normal points -Z).
    ba, bb, bc, bd = a + bot_start, b + bot_start, c + bot_start, d + bot_start
    bot_faces = np.vstack([np.stack([ba, bc, bb], 1), np.stack([ba, bd, bc], 1)])

    # Side walls — stitch all four perimeters so the mesh is watertight.
    j = np.arange(w - 1)
    i = np.arange(h - 1)
    back_row = (h - 1) * w

    # Front (i=0, -Y) and back (i=h-1, +Y) walls.
    front = np.vstack([
        np.stack([j, j + 1, bot_start + j + 1], 1),
        np.stack([j, bot_start + j + 1, bot_start + j], 1),
    ])
    back = np.vstack([
        np.stack([back_row + j, bot_start + back_row + j + 1, back_row + j + 1], 1),
        np.stack([back_row + j, bot_start + back_row + j, bot_start + back_row + j + 1], 1),
    ])

    # Left (j=0, -X) and right (j=w-1, +X) walls.
    lt0, lt1 = i * w, (i + 1) * w
    left = np.vstack([
        np.stack([lt0, bot_start + lt0, bot_start + lt1], 1),
        np.stack([lt0, bot_start + lt1, lt1], 1),
    ])
    rt0, rt1 = i * w + (w - 1), (i + 1) * w + (w - 1)
    right = np.vstack([
        np.stack([rt0, rt1, bot_start + rt1], 1),
        np.stack([rt0, bot_start + rt1, bot_start + rt0], 1),
    ])

    groups = [top_faces, bot_faces, front, back, left, right]
    faces = np.vstack(groups)

    # Orient every face outward. trimesh's fix_normals() would do this, but it
    # runs an adjacency-graph winding traversal that takes minutes on a ~2M-face
    # relief (the app looks hung). Because this is a heightfield extrusion we
    # know each face group's outward direction analytically, so we flip mis-wound
    # faces in one vectorized O(n) pass instead: compute each triangle's geometric
    # normal and reverse the winding of any whose normal points inward.
    outward = np.vstack([
        np.tile(d, (len(g), 1))
        for g, d in zip(groups, [(0, 0, 1), (0, 0, -1),   # top +Z, bottom -Z
                                 (0, -1, 0), (0, 1, 0),    # front -Y, back +Y
                                 (-1, 0, 0), (1, 0, 0)])   # left -X, right +X
    ])
    tris = vertices[faces]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    flip = (normals * outward).sum(1) < 0
    faces[flip] = faces[flip][:, ::-1]

    # process=False: skip trimesh's automatic (and slow) merge/winding pass — the
    # walls already share the top/bottom vertices, so the mesh is watertight as
    # built and the winding is now correct.
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.merge_vertices()  # weld any coincident points for a clean manifold

    return mesh, (scaled_w, scaled_d, scaled_h, scale)


def generate_preview_mesh(z_grid, downsample_factor=4):
    """
    Generate low-res mesh for live preview (faster).
    """
    downsampled = z_grid[::downsample_factor, ::downsample_factor]
    # Normalize again after downsampling
    downsampled = normalize_to_01(downsampled)
    return downsampled


def mesh_to_stl_bytes(mesh):
    """Convert trimesh object to STL bytes."""
    if trimesh is None:
        raise ImportError("trimesh not installed")

    buf = io.BytesIO()
    # Binary STL: ~50 bytes/triangle vs. ~250 for ASCII. On a large relief the
    # ASCII text buffer alone can reach multiple GB and OOM the worker; binary
    # keeps it manageable and exports faster. Slicers read both identically.
    mesh.export(buf, file_type='stl')
    buf.seek(0)
    return buf.read()
