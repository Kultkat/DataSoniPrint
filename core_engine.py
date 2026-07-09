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
    import mpmath  # ships with sympy; used for complex special functions (zeta, gamma)
except ImportError:
    mpmath = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import trimesh
except ImportError:
    trimesh = None


def memory_usage_mb():
    """
    Current resident memory (RSS) of this process, in MB. Used to show the user
    how close the app is to its RAM budget. Reads Linux /proc (Streamlit Cloud is
    Linux); falls back to the peak from ``resource`` elsewhere. Returns 0.0 if
    neither is available. NOTE: a hard out-of-memory kill (SIGKILL from the OS)
    cannot be reported by the process itself — this only helps you watch the
    trend and catch Python-level MemoryErrors.
    """
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0


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

def evaluate_formula(formula_str, x_range=(-5, 5), y_range=(-5, 5), resolution=50,
                     complex_mode=False, projection="abs",
                     clip_percent=None, log_scale=False):
    """
    Evaluate a formula on a grid and return a normalized 2D height array [0, 1].

    Two modes:
      - real (default): ``z = f(x, y)`` over the real plane, e.g. 'sin(x)*cos(y)'.
      - complex (``complex_mode=True``): the variable is ``z = x + i·y`` and the
        height is a real *projection* of f(z) — one of 'abs', 're', 'im', 'phase'.
        This is how complex functions (Riemann zeta, gamma, polynomials) become
        printable surfaces. Complex evaluation uses mpmath, so special functions
        like ``zeta(z)`` and ``gamma(z)`` work.

    Printability helpers (functions with poles spike to infinity otherwise):
      - ``clip_percent``: clip the field to its [p, 100-p] percentiles (e.g. 1).
      - ``log_scale``: compress with log1p before normalizing (tames sharp peaks).
    """
    if sp is None:
        raise ImportError("sympy not installed")

    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    xx, yy = np.meshgrid(x_vals, y_vals)

    if complex_mode:
        zz = _evaluate_complex(formula_str, xx, yy, projection)
    else:
        x, y = symbols('x y', real=True)
        try:
            # Bind x/y so the parsed expression uses OUR symbols (not fresh ones).
            expr = sympify(formula_str, locals={"x": x, "y": y})
        except Exception as e:
            raise ValueError(f"Invalid formula: {e}")
        try:
            f = sp.lambdify((x, y), expr, modules=["numpy"])
            zz = np.asarray(f(xx, yy), dtype=float)
            if zz.shape != xx.shape:  # formula independent of x and/or y → scalar
                zz = np.broadcast_to(zz, xx.shape).astype(float)
        except Exception as e:
            raise ValueError(f"Could not evaluate formula: {e}")

    # Drop non-finite values (e.g. poles, log of negatives) before scaling.
    zz = np.where(np.isfinite(zz), zz, np.nan)

    if clip_percent:
        lo = np.nanpercentile(zz, clip_percent)
        hi = np.nanpercentile(zz, 100 - clip_percent)
        zz = np.clip(zz, lo, hi)

    if log_scale:
        zz = np.log1p(zz - np.nanmin(zz))

    # Normalize to [0, 1]
    return normalize_to_01(zz)


def _evaluate_complex(formula_str, xx, yy, projection="abs"):
    """
    Evaluate a complex function f(z) with z = x + i·y over the grid and return a
    real field via ``projection`` ('abs' | 're' | 'im' | 'phase').
    """
    if mpmath is None:
        raise ImportError("mpmath not installed (ships with sympy)")

    z = symbols('z')
    try:
        expr = sympify(formula_str, locals={"z": z})
    except Exception as e:
        raise ValueError(f"Invalid complex formula: {e}")
    try:
        f = sp.lambdify(z, expr, modules=["mpmath"])
    except Exception as e:
        raise ValueError(f"Could not compile complex formula: {e}")

    proj = {
        "abs": lambda w: abs(w),
        "re": lambda w: w.real,
        "im": lambda w: w.imag,
        "phase": lambda w: np.angle(w),
    }.get(projection, lambda w: abs(w))

    def _point(c):
        try:
            return float(proj(complex(f(complex(c)))))
        except Exception:
            return np.nan

    grid = xx + 1j * yy
    return np.vectorize(_point, otypes=[float])(grid)


def mandelbrot_field(x_range=(-2.0, 0.6), y_range=(-1.3, 1.3),
                     resolution=200, max_iter=120):
    """
    Mandelbrot escape-time field, normalized to [0, 1] — height = how quickly the
    point z→z²+c diverges. A printable fractal that needs no closed-form formula.
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    C = x[None, :] + 1j * y[:, None]
    Z = np.zeros_like(C)
    out = np.full(C.shape, float(max_iter))
    alive = np.ones(C.shape, dtype=bool)
    for i in range(max_iter):
        Z[alive] = Z[alive] ** 2 + C[alive]
        escaped = alive & (np.abs(Z) > 2.0)
        out[escaped] = i
        alive &= ~escaped
    return normalize_to_01(out)


# ─────────────────────────────────────────────────────────────────────────────
# Equation Gallery — curated famous equations, each reduced to a printable
# heightfield z(x, y). Every entry carries the pretty equation (LaTeX), a
# plain-language blurb, and sensible defaults so it renders well out of the box.
# type: "real" (z=f(x,y)) | "complex" (height = projection of f(x+iy)) | "mandelbrot".
# ─────────────────────────────────────────────────────────────────────────────
EQUATION_GALLERY = [
    {
        "key": "riemann_zeta", "name": "Riemann zeta  |ζ(s)|", "type": "complex",
        "formula": "zeta(z)", "projection": "abs",
        "latex": r"\zeta(s)=\sum_{n=1}^{\infty}\frac{1}{n^{s}}",
        "blurb": "The function at the heart of the Riemann Hypothesis. Plotted over "
                 "the complex plane, its height dips to zero along the 'critical "
                 "line' (Re = ½) — the famous non-trivial zeros that encode the "
                 "distribution of the prime numbers.",
        "x_range": (-3.0, 5.0), "y_range": (0.0, 35.0), "resolution": 130,
        "height_mm": 25.0, "clip_percent": 1.0, "log_scale": True,
    },
    {
        "key": "gamma", "name": "Gamma function  |Γ(z)|", "type": "complex",
        "formula": "gamma(z)", "projection": "abs",
        "latex": r"\Gamma(z)=\int_{0}^{\infty} t^{\,z-1}e^{-t}\,dt",
        "blurb": "The factorial generalized to all numbers: Γ(n) = (n−1)!. Over the "
                 "complex plane it rises into sharp poles at zero and the negative "
                 "integers — a dramatic, spiky landscape.",
        "x_range": (-4.0, 4.0), "y_range": (-3.0, 3.0), "resolution": 130,
        "height_mm": 25.0, "clip_percent": 2.0, "log_scale": True,
    },
    {
        "key": "particle_in_box", "name": "Schrödinger — particle in a box", "type": "real",
        "formula": "sin(2*pi*x)*sin(3*pi*y)",
        "latex": r"\hat{H}\psi = E\psi,\quad \psi_{nm}=\sin(n\pi x)\sin(m\pi y)",
        "blurb": "A quantum particle trapped in a 2D box can only occupy discrete "
                 "standing-wave states. This is the (n=2, m=3) state — the kind of "
                 "solution Schrödinger's equation produces.",
        "x_range": (0.0, 1.0), "y_range": (0.0, 1.0), "resolution": 120,
        "height_mm": 18.0,
    },
    {
        "key": "hydrogen_orbital", "name": "Hydrogen orbital (2p)", "type": "real",
        "formula": "x*exp(-sqrt(x**2+y**2)/2)",
        "latex": r"\psi_{2p}\;\propto\; r\,e^{-r/2}\cos\theta",
        "blurb": "The dumbbell-shaped electron probability cloud of a hydrogen atom's "
                 "2p orbital — one of the iconic images of quantum chemistry.",
        "x_range": (-10.0, 10.0), "y_range": (-10.0, 10.0), "resolution": 120,
        "height_mm": 20.0,
    },
    {
        "key": "wave_packet", "name": "Quantum wave packet", "type": "real",
        "formula": "exp(-(x**2+y**2)/4)*cos(3*x)",
        "latex": r"\psi(x)=e^{-x^{2}/4}\,e^{ikx}",
        "blurb": "A particle that is also a wave: a localized Gaussian envelope "
                 "wrapped around an oscillation. The bridge between particle and "
                 "wave pictures in quantum mechanics.",
        "x_range": (-6.0, 6.0), "y_range": (-6.0, 6.0), "resolution": 120,
        "height_mm": 20.0,
    },
    {
        "key": "gravity_well", "name": "Gravity well (−GM/r)", "type": "real",
        "formula": "-1/sqrt(x**2+y**2+0.05)",
        "latex": r"\Phi(r)=-\frac{GM}{r}",
        "blurb": "The classic 'bowling ball on a rubber sheet' picture: the deeper "
                 "the well, the stronger gravity pulls. A planet's potential as a "
                 "literal dip in space.",
        "x_range": (-5.0, 5.0), "y_range": (-5.0, 5.0), "resolution": 120,
        "height_mm": 22.0, "clip_percent": 1.0,
    },
    {
        "key": "flamm_paraboloid", "name": "Black-hole geometry (Flamm)", "type": "real",
        "formula": "2*sqrt(sqrt(x**2+y**2)-1)",
        "latex": r"z(r)=2\sqrt{r_{s}}\,\sqrt{r-r_{s}}",
        "blurb": "Flamm's paraboloid — the curved shape of space just outside a "
                 "Schwarzschild black hole's event horizon. The funnel everyone "
                 "pictures, drawn straight from General Relativity.",
        "x_range": (-6.0, 6.0), "y_range": (-6.0, 6.0), "resolution": 140,
        "height_mm": 22.0,
    },
    {
        "key": "mandelbrot", "name": "Mandelbrot set", "type": "mandelbrot",
        "formula": "z_{n+1} = z_n^2 + c",
        "latex": r"z_{n+1}=z_{n}^{2}+c",
        "blurb": "The most famous fractal. A dead-simple rule repeated forever "
                 "produces infinite detail at its boundary — height here shows how "
                 "fast each point escapes to infinity.",
        "x_range": (-2.0, 0.6), "y_range": (-1.3, 1.3), "resolution": 220,
        "height_mm": 15.0,
    },
    {
        "key": "sinc_ripple", "name": "Diffraction ripple (sinc)", "type": "real",
        "formula": "sin(sqrt(x**2+y**2+1e-9))/sqrt(x**2+y**2+1e-9)",
        "latex": r"\mathrm{sinc}(r)=\frac{\sin r}{r}",
        "blurb": "Concentric ripples like a stone dropped in water, or light "
                 "diffracting through a circular aperture — the sinc function.",
        "x_range": (-15.0, 15.0), "y_range": (-15.0, 15.0), "resolution": 140,
        "height_mm": 18.0,
    },
    {
        "key": "monkey_saddle", "name": "Monkey saddle", "type": "real",
        "formula": "x**3 - 3*x*y**2",
        "latex": r"z=x^{3}-3xy^{2}",
        "blurb": "A saddle with three downward slopes instead of two — room for two "
                 "legs and a tail. A favorite example from multivariable calculus.",
        "x_range": (-2.0, 2.0), "y_range": (-2.0, 2.0), "resolution": 120,
        "height_mm": 20.0,
    },
    {
        "key": "soliton", "name": "Soliton (sech²)", "type": "real",
        "formula": "1/cosh(sqrt(x**2+y**2))**2",
        "latex": r"u(x,t)=\mathrm{sech}^{2}\!\left(\tfrac{x-ct}{2}\right)",
        "blurb": "A solitary wave that holds its shape as it travels — solitons "
                 "appear in shallow water, optical fibers, and the KdV equation.",
        "x_range": (-6.0, 6.0), "y_range": (-6.0, 6.0), "resolution": 120,
        "height_mm": 20.0,
    },
    {
        "key": "ricker_wavelet", "name": "Ricker wavelet (Mexican hat)", "type": "real",
        # Time-domain Ricker ψ(t) = (1 − 2π²f²t²)·e^(−π²f²t²), rendered as a 2D
        # relief by sweeping t → radius r = √(x²+y²) with dominant frequency
        # f = 1 Hz. Height is normalized, so f only sets the scale (the domain
        # below frames the main lobe + trough ring on a flat rim).
        "formula": "(1 - 2*pi**2*(x**2 + y**2)) * exp(-pi**2*(x**2 + y**2))",
        "latex": r"\psi(t)=\left(1-2\pi^{2}f^{2}t^{2}\right)e^{-\pi^{2}f^{2}t^{2}}",
        "blurb": "The Ricker wavelet — the 'Mexican hat' pulse used as the source "
                 "signature in seismic imaging and as the classic continuous "
                 "wavelet. A sharp central peak ringed by a symmetric trough. Shown "
                 "as a 2D relief: time t → radius, dominant frequency f = 1 Hz.",
        "x_range": (-0.9, 0.9), "y_range": (-0.9, 0.9), "resolution": 140,
        "height_mm": 18.0,
    },
]


def evaluate_gallery_item(item, resolution=None):
    """Render one EQUATION_GALLERY entry to a normalized [0,1] height grid."""
    res = int(resolution or item.get("resolution", 120))
    if item["type"] == "mandelbrot":
        return mandelbrot_field(item["x_range"], item["y_range"], res)
    return evaluate_formula(
        item["formula"], item["x_range"], item["y_range"], res,
        complex_mode=(item["type"] == "complex"),
        projection=item.get("projection", "abs"),
        clip_percent=item.get("clip_percent"),
        log_scale=item.get("log_scale", False),
    )


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

# OCR runs on its own higher-resolution copy of the image (independent of the
# 800 px relief cap): small labels — e.g. building codes on map pins — are far
# more legible to Tesseract when the glyphs are larger. OCR_MIN_DIM upscales
# small sources so short codes clear the recognizer's size threshold; OCR_MAX_DIM
# caps very large sources so a single OCR pass stays fast. Boxes are scaled back
# to the relief coordinate space before returning.
OCR_MIN_DIM = 1600
OCR_MAX_DIM = 2600


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


def _boxes_iou(a, b):
    """Intersection-over-union of two (x, y, w, h, ...) boxes."""
    ax, ay, aw, ah = a[0], a[1], a[2], a[3]
    bx, by, bw, bh = b[0], b[1], b[2], b[3]
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _dedupe_boxes(boxes, iou_thresh=0.4):
    """
    Collapse near-duplicate OCR reads (the same label found by more than one
    pass). ``boxes`` are ``(x, y, w, h, text, conf)``; the higher-confidence
    (then longer-text) read of each overlapping cluster is kept.
    """
    ordered = sorted(boxes, key=lambda b: (b[5], len(b[4])), reverse=True)
    kept = []
    for b in ordered:
        if any(_boxes_iou(b, k) >= iou_thresh for k in kept):
            continue
        kept.append(b)
    return kept


def detect_text_regions(image_path, min_confidence=40, psm=11, whitelist=None):
    """
    OCR the image and return word bounding boxes as
    ``[(x, y, w, h, text), ...]`` in relief-grid pixel coordinates.

    Several things make short labels on maps/plots hard for Tesseract, and all
    are handled here:

      - **Light text on a dark fill** (e.g. white building codes on coloured map
        pins). Tesseract assumes dark-on-light, so we OCR the image *and* its
        inverted copy and merge the results — the inverted pass is what recovers
        white-on-blue labels. De-duplication drops any label found by both.
      - **Scattered, isolated labels** (not flowing paragraphs). Page-seg mode
        ``11`` ("sparse text — find as much text as possible in no particular
        order") reads these far better than the default layout analysis.
      - **Small glyphs.** OCR runs on a higher-resolution copy (see
        ``OCR_MIN_DIM``/``OCR_MAX_DIM``); tiny pin labels miss entirely at the
        800 px relief size. Boxes are scaled back to the relief grid on return.
      - **Ambiguous short codes.** Pass a ``whitelist`` (e.g. capitals + digits)
        to constrain recognition to that alphabet — this makes Tesseract commit
        to codes like ``Y22``/``YG1`` instead of rejecting them, sharply raising
        recall. Leave it ``None`` for general text.

    Abbreviations (``YUS``, ``YXX``, ``YG1`` …) need no special handling: they
    are just short alphanumeric tokens, kept as long as they clear
    ``min_confidence`` and contain a letter or digit.

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

    # Boxes are returned in the relief coordinate space (<= MAX_RELIEF_DIM) so
    # they line up with build_image_relief / apply_labels.
    relief_img = _load_grayscale(image_path)
    rw, rh = relief_img.size

    # OCR on a higher-resolution copy; upscale small sources so short labels are
    # legible, then map the resulting boxes back to the relief grid.
    ocr_img = _load_grayscale(image_path, max_dim=OCR_MAX_DIM)
    if max(ocr_img.size) < OCR_MIN_DIM:
        f = OCR_MIN_DIM / max(ocr_img.size)
        ocr_img = ocr_img.resize(
            (max(1, round(ocr_img.width * f)), max(1, round(ocr_img.height * f))),
            Image.LANCZOS,
        )
    ow, oh = ocr_img.size
    sx, sy = rw / ow, rh / oh  # OCR pixels → relief pixels

    arr = np.array(ocr_img, dtype=np.uint8)
    # Pass 1: as-is (dark text on light — plot axes, black map labels).
    # Pass 2: inverted (light text on dark — white labels on coloured pins).
    variants = (arr, 255 - arr)
    config = f"--psm {int(psm)}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"

    collected = []
    for variant in variants:
        try:
            data = pytesseract.image_to_data(
                Image.fromarray(variant),
                output_type=pytesseract.Output.DICT,
                config=config,
            )
        except Exception as e:  # TesseractNotFoundError and friends
            raise RuntimeError(
                "Tesseract OCR engine not found. Install it with "
                "`sudo apt-get install tesseract-ocr` (Linux) or "
                "`brew install tesseract` (macOS)."
            ) from e
        for i in range(len(data["text"])):
            txt = (data["text"][i] or "").strip()
            if not txt or not any(c.isalnum() for c in txt):
                continue
            try:
                conf = float(data["conf"][i])
            except (ValueError, TypeError):
                conf = -1.0
            if conf < min_confidence:
                continue
            collected.append((
                int(round(data["left"][i] * sx)), int(round(data["top"][i] * sy)),
                int(round(data["width"][i] * sx)), int(round(data["height"][i] * sy)),
                txt, conf,
            ))

    # Merge the two passes and drop the confidence field to keep the public
    # (x, y, w, h, text) contract.
    return [(b[0], b[1], b[2], b[3], b[4]) for b in _dedupe_boxes(collected)]


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
    Digits are preceded by the number sign. Capitalization follows the standard
    indicators: a single capital sign before one capital letter, and a doubled
    capital sign (the "capital word" indicator) before a run of two or more —
    so all-caps abbreviations like ``YUS`` render compactly as ⠠⠠⠽⠥⠎ rather
    than repeating the sign before every letter.
    """
    cells = []
    in_number = False
    caps_word = False  # a capital-word indicator is currently in effect
    n = len(text)
    for i, ch in enumerate(text):
        if ch == " ":
            cells.append(())
            in_number = False
            caps_word = False
            continue
        if ch.isalpha():
            if ch.isupper():
                if not caps_word:
                    # Look ahead: 2+ consecutive capitals → capital-word sign.
                    run = 0
                    j = i
                    while j < n and text[j].isalpha() and text[j].isupper():
                        run += 1
                        j += 1
                    if run >= 2:
                        cells.append(_BRAILLE_CAPITAL_SIGN)
                        cells.append(_BRAILLE_CAPITAL_SIGN)
                        caps_word = True
                    else:
                        cells.append(_BRAILLE_CAPITAL_SIGN)
                low = ch.lower()
            else:
                caps_word = False
                low = ch
            in_number = False
            if low in _BRAILLE_LETTERS:
                cells.append(_BRAILLE_LETTERS[low])
            continue
        if ch.isdigit():
            if not in_number:
                cells.append(_BRAILLE_NUMBER_SIGN)
                in_number = True
            cells.append(_BRAILLE_LETTERS[_BRAILLE_DIGITS[ch]])
            continue
        in_number = False
        caps_word = False
        if ch in _BRAILLE_PUNCT:
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
                 dot_pitch_mm=2.5, cell_pitch_mm=6.0,
                 fit_to_box=False, min_dot_dia_mm=0.4):
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

    With ``fit_to_box=True`` each label's braille cluster is scaled down (keeping
    the standard 1.5 : 2.5 : 6.0 dot/pitch/cell proportions) to span the original
    text box, so it occupies the same footprint as the printed label — useful for
    dense maps where full-size cells would overlap. Dots never grow beyond the
    standard size and never shrink below ``min_dot_dia_mm`` (kept printable).
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
        value = dot_height_mm / height_mm
        # dot number → (column, row) within the cell
        dot_pos = {1: (0, 0), 2: (0, 1), 3: (0, 2),
                   4: (1, 0), 5: (1, 1), 6: (1, 2)}
        # Standard proportions, reused when fitting so the cluster stays braille-
        # shaped as it scales: cluster width = (n-1)·cell + dot_pitch + dot_dia.
        r_dp = dot_pitch_mm / cell_pitch_mm
        r_dd = dot_dia_mm / cell_pitch_mm
        for box in (text_boxes or []):
            x, y, bw, bh = box[0], box[1], box[2], box[3]
            text = box[4] if len(box) > 4 else ""
            cells = text_to_braille_cells(text)
            if not cells:
                continue
            n = len(cells)

            cell_pitch, dot_pitch, dot_dia = cell_pitch_mm, dot_pitch_mm, dot_dia_mm
            if fit_to_box:
                box_w_mm = max(bw * mmx, 1e-6)
                denom = (n - 1) + r_dp + r_dd
                cp = box_w_mm / denom if denom > 0 else cell_pitch_mm
                cell_pitch = min(cp, cell_pitch_mm)   # never larger than standard
                dot_pitch = cell_pitch * r_dp
                dot_dia = max(cell_pitch * r_dd, min_dot_dia_mm)

            rx = (dot_dia / 2.0) / mmx
            ry = (dot_dia / 2.0) / mmy
            # Centre the whole cluster on the box (works for point-marker labels
            # as well as axis text).
            cluster_w_mm = (n - 1) * cell_pitch + dot_pitch + dot_dia
            box_cx_mm = (x + bw / 2.0) * mmx
            box_cy_mm = (y + bh / 2.0) * mmy
            start_x_mm = box_cx_mm - cluster_w_mm / 2.0 + dot_dia / 2.0
            top_y_mm = box_cy_mm - dot_pitch  # middle of the 3 rows on box centre
            for ci, cell in enumerate(cells):
                cell_x_mm = start_x_mm + ci * cell_pitch
                for dot in cell:
                    col, row = dot_pos[dot]
                    cx_mm = cell_x_mm + col * dot_pitch
                    cy_mm = top_y_mm + row * dot_pitch
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
