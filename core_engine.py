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
        expr = sympify(formula_str)
    except Exception as e:
        raise ValueError(f"Invalid formula: {e}")

    # Generate grid
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    xx, yy = np.meshgrid(x_vals, y_vals)

    # Evaluate
    zz = np.zeros_like(xx)
    for i in range(resolution):
        for j in range(resolution):
            try:
                val = float(expr.subs({x: xx[i, j], y: yy[i, j]}))
                zz[i, j] = val
            except:
                zz[i, j] = 0.0

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

def image_to_relief_array(image_path, height_min_mm=0.5, height_max_mm=5.0):
    """
    Convert image to grayscale, map intensity (0-255) to height (height_min to height_max).
    Returns normalized 2D array.
    """
    if Image is None:
        raise ImportError("Pillow not installed")

    img = Image.open(image_path).convert('L')  # Grayscale
    img_array = np.array(img, dtype=np.float64)

    # Normalize to [0, 1]
    img_norm = img_array / 255.0

    return img_norm


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

    # Build mesh
    x_coords = np.linspace(0, scaled_w, w)
    y_coords = np.linspace(0, scaled_d, h)

    vertices = []
    faces = []

    # Top surface
    for i in range(h):
        for j in range(w):
            z_val = 1.0 + z_grid[i, j] * scaled_h  # 1mm base + scaled height
            vertices.append([x_coords[j], y_coords[i], z_val])

    # Bottom surface (1mm base)
    for i in range(h):
        for j in range(w):
            vertices.append([x_coords[j], y_coords[i], 0.0])

    # Top faces
    for i in range(h - 1):
        for j in range(w - 1):
            v0 = i * w + j
            v1 = i * w + (j + 1)
            v2 = (i + 1) * w + (j + 1)
            v3 = (i + 1) * w + j

            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    # Bottom faces
    bot_start = h * w
    for i in range(h - 1):
        for j in range(w - 1):
            v0 = bot_start + i * w + j
            v1 = bot_start + i * w + (j + 1)
            v2 = bot_start + (i + 1) * w + (j + 1)
            v3 = bot_start + (i + 1) * w + j

            faces.append([v0, v2, v1])
            faces.append([v0, v3, v2])

    # Side faces
    for j in range(w - 1):
        v0_t = j
        v1_t = j + 1
        v0_b = bot_start + j
        v1_b = bot_start + j + 1
        faces.append([v0_t, v1_t, v1_b])
        faces.append([v0_t, v1_b, v0_b])

    for i in range(h - 1):
        v0_t = i * w
        v1_t = (i + 1) * w
        v0_b = bot_start + i * w
        v1_b = bot_start + (i + 1) * w
        faces.append([v0_t, v0_b, v1_b])
        faces.append([v0_t, v1_b, v1_t])

    # Create trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.remove_duplicate_faces()
    mesh.merge_vertices()

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
    mesh.export(buf, file_type='stl_ascii')
    buf.seek(0)
    return buf.read()
