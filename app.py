"""
DataSoniPrint Streamlit App — Phase 1-4 Implementation

A clean, Python-native interface for converting 2D data into 3D-printable relief STLs.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import io
import re
import tempfile

from core_engine import (
    load_file, normalize_to_01, select_columns_for_axes,
    evaluate_formula, generate_preview_mesh, scale_to_bed, mesh_to_stl_bytes,
    image_to_relief_array, spread_to_grid,
    detect_text_regions, build_image_relief, apply_labels,
    EQUATION_GALLERY, evaluate_gallery_item, memory_usage_mb,
)


def _bed_from_domain(xspan, yspan):
    """Bed (width_mm, depth_mm) matching a domain's aspect ratio: longer side
    fills the 200mm bed, the other is scaled to match (rounded to the 10mm slider
    step). Keeps formula/gallery reliefs from being squished into a square."""
    xspan, yspan = abs(xspan) or 1.0, abs(yspan) or 1.0
    if xspan >= yspan:
        return 200, max(10, min(200, round(200 * yspan / xspan / 10) * 10))
    return max(10, min(200, round(200 * xspan / yspan / 10) * 10)), 200

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DataSoniPrint",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎨 DataSoniPrint — 2D Data → 3D Relief STL")
st.markdown("""
Convert any CSV, HDF5, NetCDF, GRIB, or ASDF file into a 3D-printable relief model.
Upload data, configure scaling, and download STL for your 3D printer.
""")

# Live memory readout — lets you see how close the app is to its RAM budget.
# (A hard out-of-memory kill can't print anything, so this trend is the warning.)
st.sidebar.metric("🧠 Memory in use", f"{memory_usage_mb():.0f} MB")
st.sidebar.caption(
    "Hosted apps have a fixed RAM budget. If memory climbs steeply and the app "
    "then freezes or reloads, it ran out of memory — try a smaller file."
)

# ─────────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────────

if "current_z_grid" not in st.session_state:
    st.session_state.current_z_grid = None
if "current_mesh" not in st.session_state:
    st.session_state.current_mesh = None
if "mesh_dims" not in st.session_state:
    st.session_state.mesh_dims = None
if "loaded_file" not in st.session_state:
    st.session_state.loaded_file = None
if "columns" not in st.session_state:
    st.session_state.columns = None
if "headers" not in st.session_state:
    st.session_state.headers = None
if "height_mm" not in st.session_state:
    st.session_state.height_mm = 20.0
if "width_mm" not in st.session_state:
    st.session_state.width_mm = 150
if "depth_mm" not in st.session_state:
    st.session_state.depth_mm = 100
if "label_spec" not in st.session_state:
    st.session_state.label_spec = None  # image-mode axis-label rendering plan
if "source_name" not in st.session_state:
    st.session_state.source_name = "relief"  # stem of the source, for the STL filename


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Upload (one uploader for all formats — the mode auto-detects below)
# ─────────────────────────────────────────────────────────────────────────────

DATA_EXTS = {".csv", ".h5", ".hdf5", ".hdf", ".nc", ".nc4", ".netcdf",
             ".grib", ".grib2", ".grb", ".grb2", ".asdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

st.header("📥 Step 1: Upload Data or Image")

uploaded_file = st.file_uploader(
    "Upload a data file (CSV/HDF5/NetCDF/GRIB/ASDF) or an image (PNG/JPG) — "
    "or skip this and pick 'From Math Formula' below:",
    type=["csv", "h5", "hdf5", "hdf", "nc", "nc4", "netcdf",
          "grib", "grib2", "grb", "grb2", "asdf", "png", "jpg", "jpeg"],
)

uploaded_ext = Path(uploaded_file.name).suffix.lower() if uploaded_file else None

# Auto-select the mode from the file extension whenever a NEW file is uploaded.
# (Manual radio changes are preserved until a different file is uploaded.)
if uploaded_file is not None and st.session_state.get("last_uploaded") != uploaded_file.name:
    st.session_state.last_uploaded = uploaded_file.name
    st.session_state.mode = (
        "🖼️ From Image" if uploaded_ext in IMAGE_EXTS else "📊 From Data Column"
    )

if "mode" not in st.session_state:
    st.session_state.mode = "📊 From Data Column"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Mode Selection (auto-set from the upload; override here or pick Formula)
# ─────────────────────────────────────────────────────────────────────────────

st.header("🔄 Step 2: Select Mode")

mode = st.radio(
    "How would you like to generate the relief? "
    "(auto-set from your upload — change it for a formula or to override)",
    ["📊 From Data Column", "🔢 From Math Formula", "🖼️ From Image"],
    key="mode",
)

# ─────────────────────────────────────────────────────────────────────────────
# MODE 1: Data Column
# ─────────────────────────────────────────────────────────────────────────────

if mode == "📊 From Data Column":
    if uploaded_file is not None and uploaded_ext in DATA_EXTS:
        try:
            size_mb = getattr(uploaded_file, "size", 0) / 1e6
            mem_before = memory_usage_mb()
            if size_mb > 100:
                st.warning(
                    f"⚠️ Large data file ({size_mb:.0f} MB). Loading it can use "
                    "several times its size in RAM. If the app freezes or reloads "
                    "right after this, it ran out of memory — see the memory gauge "
                    "in the sidebar."
                )
            with st.spinner(f"Loading {uploaded_file.name} ({size_mb:.0f} MB)…"):
                file_bytes = uploaded_file.read()
                headers, columns = load_file(file_bytes, uploaded_file.name)
                del file_bytes  # free the raw bytes once parsed

            mem_after = memory_usage_mb()
            st.caption(
                f"🧠 Memory: {mem_after:.0f} MB in use "
                f"(+{max(mem_after - mem_before, 0):.0f} MB to load this file)."
            )

            st.session_state.loaded_file = uploaded_file.name
            st.session_state.headers = headers
            st.session_state.columns = columns

            col_lengths = [len(columns[h]) for h in headers]
            len_min, len_max = min(col_lengths), max(col_lengths)
            if len_min == len_max:
                len_summary = f"{len_min:,} rows"
            else:
                len_summary = f"{len_min:,}–{len_max:,} rows (columns differ in length)"
            st.success(f"✓ Loaded {uploaded_file.name} — {len(headers)} columns, {len_summary}")

            # Show column stats
            with st.expander("📈 Column Statistics"):
                col1, col2 = st.columns(2)
                for idx, header in enumerate(headers):
                    data = columns[header]
                    if idx % 2 == 0:
                        with col1:
                            st.metric(f"**{header}**", f"{len(data)} points")
                            st.caption(f"Min: {data.min():.3g} | Max: {data.max():.3g} | Mean: {data.mean():.3g}")
                    else:
                        with col2:
                            st.metric(f"**{header}**", f"{len(data)} points")
                            st.caption(f"Min: {data.min():.3g} | Max: {data.max():.3g} | Mean: {data.mean():.3g}")

        except MemoryError:
            st.error(
                f"❌ Out of memory while loading this file "
                f"({memory_usage_mb():.0f} MB in use). It's too big for the app's "
                "RAM budget — try a smaller file, or downsample/trim it first."
            )
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
    elif uploaded_file is not None and uploaded_ext in IMAGE_EXTS:
        st.warning("That's an image — switch to '🖼️ From Image' mode above (Step 2).")

    if st.session_state.headers:
        st.subheader("Select Columns for X, Y, Z Axes")

        def _col_label(h):
            n = len(st.session_state.columns[h])
            return f"{h}  ({n:,} samples)"

        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox(
                "X-axis (width):",
                st.session_state.headers,
                format_func=_col_label,
                key="x_col",
            )
        with col2:
            y_col = st.selectbox(
                "Y-axis (depth):",
                st.session_state.headers,
                index=min(1, len(st.session_state.headers) - 1),
                format_func=_col_label,
                key="y_col",
            )
        with col3:
            z_col = st.selectbox(
                "Z-axis (height):",
                st.session_state.headers,
                index=min(2, len(st.session_state.headers) - 1),
                format_func=_col_label,
                key="z_col",
            )

        st.markdown("**Spread** — how to project 1D column data onto a 2D relief grid:")
        col1, col2 = st.columns(2)
        with col1:
            spread_mode = st.radio(
                "Spread mode:",
                ["reshape", "scatter"],
                horizontal=True,
                help=(
                    "reshape: resample Z and fold it into a square grid (best for "
                    "time-series like GW strain). scatter: interpolate (X, Y, Z) "
                    "triples onto a regular grid (needs distinct X and Y)."
                ),
            )
        with col2:
            spread_res = st.slider("Spread resolution:", 32, 256, 100, step=8, key="spread_res")

        if st.button("▶ Generate from Data", key="gen_data"):
            with st.spinner("Spreading data into a 2D relief grid..."):
                try:
                    x_data, y_data, z_data = select_columns_for_axes(
                        st.session_state.columns,
                        st.session_state.headers,
                        x_col, y_col, z_col
                    )

                    z_grid = spread_to_grid(
                        x_data, y_data, z_data,
                        resolution=spread_res,
                        mode=spread_mode,
                    )

                    st.session_state.current_z_grid = z_grid
                    st.session_state.label_spec = None
                    st.session_state.source_name = Path(
                        st.session_state.loaded_file or "data"
                    ).stem
                    st.success(
                        f"✓ Spread {len(z_data)} Z samples → {z_grid.shape[0]}×{z_grid.shape[1]} relief grid"
                    )
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    else:
        st.info("⬆️ Upload a data file above to choose columns and generate a relief.")

# ─────────────────────────────────────────────────────────────────────────────
# MODE 2: Formula
# ─────────────────────────────────────────────────────────────────────────────

elif mode == "🔢 From Math Formula":
    st.header("🔢 Equations & Formulas")

    GALLERY_NAMES = ["✏️ Custom formula"] + [it["name"] for it in EQUATION_GALLERY]
    choice = st.selectbox(
        "Pick a famous equation, or write your own:",
        GALLERY_NAMES,
        help="Curated equations come with an explanation and printable defaults. "
             "Choose '✏️ Custom formula' to type any expression yourself.",
    )
    selected = (None if choice == GALLERY_NAMES[0]
                else EQUATION_GALLERY[GALLERY_NAMES.index(choice) - 1])

    # ── Curated gallery equation ─────────────────────────────────────────────
    if selected:
        st.latex(selected["latex"])
        st.info(selected["blurb"])
        if selected["type"] == "complex":
            st.caption(
                f"Complex function over z = x + i·y — height shows the "
                f"**{selected['projection']}** projection of f(z). "
                "This is how complex functions become 3D surfaces."
            )
        elif selected["type"] == "mandelbrot":
            st.caption("Iterated map z → z² + c — height shows how fast each point escapes.")
        st.code(selected["formula"])

        default_res = int(selected.get("resolution", 120))
        resolution = st.slider("Detail (grid resolution):", 40, 260,
                               min(default_res, 260), step=10)
        if selected["type"] == "complex":
            st.caption("⏳ Complex equations are evaluated point-by-point and capped "
                       "at 150 for speed — Riemann ζ takes a few seconds.")

        if st.button("▶ Generate this equation", key="gen_gallery"):
            with st.spinner(f"Rendering {selected['name']}…"):
                try:
                    res = (min(resolution, 150) if selected["type"] == "complex"
                           else resolution)
                    z_grid = evaluate_gallery_item(selected, resolution=res)
                    st.session_state.current_z_grid = z_grid
                    st.session_state.label_spec = None
                    st.session_state.source_name = f"equation_{selected['key']}"
                    st.session_state.height_mm = float(selected.get("height_mm", 20.0))
                    # Match the bed to the equation's domain so it isn't squished.
                    bw, bd = _bed_from_domain(
                        selected["x_range"][1] - selected["x_range"][0],
                        selected["y_range"][1] - selected["y_range"][0],
                    )
                    st.session_state.width_mm, st.session_state.depth_mm = bw, bd
                    st.success(f"✓ {selected['name']} rendered")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    # ── Custom free-form formula ─────────────────────────────────────────────
    else:
        formula = st.text_input(
            "Formula (e.g., 'sin(x)*cos(y)', 'exp(-x**2 - y**2)', 'zeta(z)'):",
            value="sin(x)*cos(y)",
            help=(
                "Enter the right-hand side only — no 'z =' or 'f(x,y) ='. "
                "Real mode uses variables x and y; complex mode uses z = x + i·y."
            ),
        )

        complex_mode = st.checkbox(
            "Complex function f(z), with z = x + i·y",
            help="Treats the input as a function of a complex variable and uses a "
                 "real projection as height — this is how Riemann zeta, gamma and "
                 "other complex functions become surfaces. Special functions like "
                 "zeta(z) and gamma(z) are available in this mode.",
        )
        projection = "abs"
        if complex_mode:
            projection = st.selectbox(
                "Height shows:",
                ["abs", "re", "im", "phase"],
                format_func=lambda p: {
                    "abs": "|f(z)| — magnitude",
                    "re": "Re f(z) — real part",
                    "im": "Im f(z) — imaginary part",
                    "phase": "arg f(z) — phase",
                }[p],
            )

        with st.expander("ℹ️ Formula format — what works and what doesn't"):
            st.markdown(
                """
**Required format**
- **Right-hand side only** — type `sin(x)*cos(y)`, not `z = sin(x)*cos(y)`.
- **Real mode:** variables are `x` and `y` (the two grid axes).
- **Complex mode:** the variable is `z = x + i·y`; tick the box above. Then
  `zeta(z)`, `gamma(z)`, `1/z`, `z**3`, etc. all work — height is the chosen
  projection (`|f|`, real, imaginary, or phase).
- **Plain ASCII math:** `*` multiply · `**` power · `/` divide · `+ -` (a normal
  hyphen-minus). Functions: `sin cos tan exp log sqrt abs`, constant `pi`.

**✅ Examples**
- `sin(x)*cos(y)`
- `exp(-x**2 - y**2)`
- `zeta(z)`  *(complex mode — the Riemann zeta function)*

**❌ Won't work**
- An assignment / left-hand side: `ψ(t) =`, `z =`
- Unicode math symbols: `·` (use `*`), `√` (use `sqrt`), `^` (use `**`),
  `−` the long minus (use `-`), `π` (use `pi`)
- Other variable names (`t`, `r`, `θ`, …) or undefined parameters (e.g. `σ`) —
  substitute a number, or rewrite in terms of `x`/`y` (e.g. radius `sqrt(x**2+y**2)`)
- Non-finite results (÷0, `log` of negatives) are dropped to the base, not raised.

The height is **normalized** to the Height-scale slider, so only the *shape*
matters — overall constants don't change the relief.
"""
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            x_range_min = st.number_input("X range (min):", value=-5.0)
            x_range_max = st.number_input("X range (max):", value=5.0)
        with col2:
            y_range_min = st.number_input("Y range (min):", value=-5.0)
            y_range_max = st.number_input("Y range (max):", value=5.0)
        with col3:
            res_max = 150 if complex_mode else 200
            resolution = st.slider("Grid Resolution:", 20, res_max,
                                   min(80, res_max), step=10)

        if st.button("▶ Generate from Formula", key="gen_formula"):
            with st.spinner("Evaluating formula and generating mesh..."):
                try:
                    z_grid = evaluate_formula(
                        formula,
                        x_range=(x_range_min, x_range_max),
                        y_range=(y_range_min, y_range_max),
                        resolution=resolution,
                        complex_mode=complex_mode,
                        projection=projection,
                        clip_percent=(1.0 if complex_mode else None),
                    )
                    st.session_state.current_z_grid = z_grid
                    st.session_state.label_spec = None
                    # Name the STL after the formula itself (sanitized to a safe,
                    # unique filename stem) so each expression downloads distinctly.
                    safe = re.sub(r"[^0-9A-Za-z]+", "_", formula).strip("_")[:40]
                    prefix = "complex" if complex_mode else "formula"
                    st.session_state.source_name = f"{prefix}_{safe}" if safe else prefix
                    # Match the bed to the formula's domain so it isn't squished.
                    bw, bd = _bed_from_domain(x_range_max - x_range_min,
                                              y_range_max - y_range_min)
                    st.session_state.width_mm, st.session_state.depth_mm = bw, bd
                    st.success("✓ Formula evaluated and mesh generated")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# MODE 3: Image
# ─────────────────────────────────────────────────────────────────────────────

elif mode == "🖼️ From Image":
    st.header("🖼️ Image Relief")

    if uploaded_file is not None and uploaded_ext in IMAGE_EXTS:
        st.success(f"✓ Using image: {uploaded_file.name}")

        invert = st.checkbox(
            "Raise data above background (invert brightness)",
            value=True,
            help=(
                "On: dark plot data becomes raised relief on a flat base. "
                "Off: bright areas become the tall regions instead."
            ),
        )
        bg_threshold = st.slider(
            "Background cutoff:", 0.0, 0.9, 0.1, step=0.05,
            help="Pixels fainter than this flatten down to the 1mm base plate.",
        )

        st.markdown("**Relief style** — how brightness/colour maps to height:")
        relief_style = st.radio(
            "Relief style:",
            ["Grayscale → height (3D relief)", "Black & white (single height)"],
            horizontal=True,
            help=(
                "Grayscale → height: each pixel's brightness/colour shade becomes "
                "a VARIABLE height on the Z axis — a full 3D relief. "
                "Black & white: all data sits at ONE flat height (a uniform step "
                "you can see and feel), regardless of shade."
            ),
        )
        BINARY = relief_style.startswith("Black")

        # Default the absolute heights; relevant slider depends on the style.
        relief_min, relief_max, bw_height = 0.1, 5.0, 0.5
        if BINARY:
            bw_height = st.slider(
                "Relief height (mm):", 0.2, 5.0, 0.5, step=0.1,
                help="The single height of every data feature above the 1mm base.",
            )
        else:
            col1, col2 = st.columns(2)
            with col1:
                relief_min = st.slider(
                    "Relief floor (mm):", 0.1, 4.0, 0.1, step=0.1,
                    help="Height of the faintest data above the base.",
                )
            with col2:
                relief_max = st.slider(
                    "Relief peak (mm):", 1.0, 10.0, 5.0, step=0.5,
                    help="Height of the tallest data above the base.",
                )

        st.markdown("**Axis labels & markers** — how to handle detected text:")
        label_mode_label = st.radio(
            "Label rendering:",
            ["Keep as relief (no OCR)", "Engrave text (OCR)", "Braille dots (OCR)"],
            horizontal=True,
            help=(
                "Keep as relief: labels stay raised as part of the data (no OCR "
                "needed). Engrave: carve labels into the plate. Braille: emboss "
                "standard tactile braille dots where each label is. OCR needs the "
                "Tesseract engine; if it's missing we fall back to 'Keep as relief'."
            ),
        )
        LABEL_MODE = {
            "Keep as relief (no OCR)": "none",
            "Engrave text (OCR)": "engrave",
            "Braille dots (OCR)": "braille",
        }[label_mode_label]

        engrave_depth = 0.3
        if LABEL_MODE == "engrave":
            engrave_depth = st.slider("Engrave depth (mm):", 0.1, 0.8, 0.3, step=0.05)
        elif LABEL_MODE == "braille":
            st.caption(
                "Braille uses standard tactile geometry (1.5mm dots, 2.5mm dot "
                "spacing, 0.6mm tall). Use a high-resolution source image and a "
                "large enough model for the dots to render cleanly."
            )

        if st.button("▶ Generate from Image", key="gen_image"):
            with st.spinner("Converting image to relief..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_ext) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    # OCR only when a label mode needs it.
                    text_boxes = []
                    effective_mode = LABEL_MODE
                    if LABEL_MODE in ("engrave", "braille"):
                        try:
                            text_boxes = detect_text_regions(tmp_path, min_confidence=40)
                            if text_boxes:
                                found = ", ".join(b[4] for b in text_boxes[:8])
                                more = "…" if len(text_boxes) > 8 else ""
                                st.info(
                                    f"🔤 OCR found {len(text_boxes)} text region(s): {found}{more}"
                                )
                            else:
                                st.info("🔤 OCR ran but found no readable text.")
                                effective_mode = "none"
                        except Exception as ocr_err:
                            st.warning(
                                f"⚠️ OCR unavailable — keeping labels as relief. ({ocr_err})"
                            )
                            effective_mode = "none"
                            text_boxes = []

                    # Build the data relief. Exclude text regions only when we're
                    # actually going to render labels separately.
                    grid, glyph = build_image_relief(
                        tmp_path,
                        invert=invert,
                        bg_threshold=bg_threshold,
                        relief_min_mm=relief_min,
                        relief_max_mm=relief_max,
                        binary=BINARY,
                        text_boxes=text_boxes if effective_mode != "none" else None,
                    )

                    # The grid is a fraction of the height scale; pick the scale
                    # that reproduces the requested absolute mm for each style.
                    height_scale = bw_height if BINARY else relief_max
                    st.session_state.current_z_grid = grid
                    st.session_state.height_mm = float(max(height_scale, 0.1))
                    st.session_state.source_name = Path(uploaded_file.name).stem

                    # Preserve the image's aspect ratio in the bed dimensions
                    # (formula mode stays square). Fit the longer side to the
                    # 200mm bed and scale the shorter to match — otherwise a
                    # non-square image is stretched onto the default 150×100mm.
                    gh, gw = grid.shape  # rows → depth (y), cols → width (x)
                    if gw >= gh:
                        bed_w = 200
                        bed_d = max(10, min(200, round(200 * gh / gw / 10) * 10))
                    else:
                        bed_d = 200
                        bed_w = max(10, min(200, round(200 * gw / gh / 10) * 10))
                    st.session_state.width_mm = int(bed_w)
                    st.session_state.depth_mm = int(bed_d)
                    st.session_state.label_spec = {
                        "mode": effective_mode,
                        "boxes": text_boxes,
                        "glyph": glyph,
                        "engrave_depth_mm": engrave_depth,
                    }

                    if BINARY:
                        style_msg = f"B/W relief at a single {bw_height:.1f}mm height"
                    else:
                        style_msg = f"data raised {relief_min:.1f}–{relief_max:.1f}mm"
                    extra = ""
                    if effective_mode == "engrave" and text_boxes:
                        extra = f", {len(text_boxes)} label(s) engraved {engrave_depth:.2f}mm"
                    elif effective_mode == "braille" and text_boxes:
                        extra = f", {len(text_boxes)} label(s) → braille"
                    st.success(
                        f"✓ Image relief ready — {style_msg} on a 1mm base{extra}"
                    )

                    Path(tmp_path).unlink()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    elif uploaded_file is not None and uploaded_ext in DATA_EXTS:
        st.warning("That's a data file — switch to '📊 From Data Column' mode above (Step 2).")
    else:
        st.info("⬆️ Upload a PNG or JPG in Step 1 to generate a relief from its brightness.")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: STL Generation & Scaling
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.current_z_grid is not None:
    st.header("⚙️ Phase 4: Configure & Export")

    col1, col2, col3 = st.columns(3)
    with col1:
        width_mm = st.slider("Width (mm):", 10, 200, step=10, key="width_mm")
    with col2:
        depth_mm = st.slider("Depth (mm):", 10, 200, step=10, key="depth_mm")
    with col3:
        # Min 0.1mm / 0.1 step so thin B/W reliefs (down to 0.2mm) set from the
        # image-mode height sliders stay within range — a value below this min
        # raised StreamlitValueBelowMinError.
        height_mm = st.slider("Height scale (mm):", 0.1, 200.0, step=0.1, key="height_mm")

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📐 Current bounding box: {width_mm}×{depth_mm}×{height_mm}mm (max 200mm per side)")
    with col2:
        smooth = st.checkbox("Apply smoothing (Gaussian blur) for easier printing")

    # Apply axis labels (image mode) now that the physical mm dimensions are
    # known — braille geometry must be sized in real millimetres.
    spec = st.session_state.label_spec
    base_grid = st.session_state.current_z_grid
    if spec and spec.get("mode") in ("engrave", "braille"):
        effective_grid = apply_labels(
            base_grid,
            spec.get("glyph"),
            spec.get("boxes"),
            mode=spec["mode"],
            width_mm=width_mm,
            depth_mm=depth_mm,
            height_mm=height_mm,
            engrave_depth_mm=spec.get("engrave_depth_mm", 0.3),
        )
    else:
        effective_grid = base_grid

    # Image rows run top→bottom. The PHYSICAL plate maps row 0 to the front
    # (y=0), mirroring the relief vertically vs. the source image, so the STL is
    # flipped to make the print match the picture. The on-screen Plotly preview,
    # however, is viewed from a camera that already mirrors that axis — so it
    # must use the UN-flipped (image-orientation) grid to read the same way as
    # the upload. (Image mode only — label_spec is set then.)
    stl_grid = np.flipud(effective_grid) if spec is not None else effective_grid

    # ─────────────────────────────────────────────────────────────────────────
    # Live Preview (low-res)
    # ─────────────────────────────────────────────────────────────────────────

    st.subheader("📱 Live 3D Preview (low resolution)")

    # Downsample for speed WITHOUT renormalizing, so engraved (negative) and
    # braille (raised) features keep their true heights in the preview. Uses the
    # image-orientation grid so the preview matches the uploaded picture.
    preview_grid = effective_grid[::4, ::4]

    # Create Plotly surface plot
    h, w = preview_grid.shape
    x = np.linspace(0, width_mm, w)
    y = np.linspace(0, depth_mm, h)
    z = preview_grid * height_mm + 1.0  # 1mm base

    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale="Viridis")])
    # The footprint (x, y) is shown in true proportion; the height (z) is visually
    # EXAGGERATED so thin reliefs (a few mm on a 100–200mm plate) don't collapse
    # into a flat pancake. z-scale still tracks the Height slider (bigger height →
    # taller preview), just floored so it's always legibly 3D. uirevision keeps
    # your camera angle across slider tweaks instead of resetting it each rerun.
    plate = max(width_mm, depth_mm, 1)
    z_exag = 4.0
    z_aspect = min(1.0, max(0.35, (height_mm / plate) * z_exag))
    fig.update_layout(
        title=f"3D Relief Preview — {width_mm}×{depth_mm}×{height_mm}mm",
        scene=dict(
            xaxis_title="Width (mm)",
            yaxis_title="Depth (mm)",
            zaxis_title="Height (mm)",
            aspectmode="manual",
            aspectratio=dict(x=width_mm / plate, y=depth_mm / plate, z=z_aspect),
            zaxis=dict(range=[0, height_mm + 1.0]),
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.2)),
            uirevision="relief",
        ),
        height=500,
        margin=dict(l=0, r=0, b=0, t=40),
        uirevision="relief",
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "ℹ️ The preview's height is exaggerated for on-screen clarity — the "
        "exported STL uses the true millimetres shown on the sliders."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Generate High-Res STL
    # ─────────────────────────────────────────────────────────────────────────

    if st.button("✨ Generate High-Res STL", key="gen_stl"):
        with st.spinner("Creating high-resolution mesh for 3D printing..."):
            try:
                mesh, dims = scale_to_bed(
                    stl_grid,
                    target_width_mm=width_mm,
                    target_depth_mm=depth_mm,
                    target_height_mm=height_mm
                )

                st.session_state.current_mesh = mesh
                st.session_state.mesh_dims = dims

                st.success("✓ STL generated successfully")
                st.info(f"📦 Final dimensions: {dims[0]:.1f}×{dims[1]:.1f}×{dims[2]:.1f}mm (scale: {dims[3]:.2f})")

            except Exception as e:
                st.error(f"❌ Error generating STL: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Download STL
    # ─────────────────────────────────────────────────────────────────────────

    if st.session_state.current_mesh is not None:
        st.subheader("💾 Download Your Relief")

        stl_bytes = mesh_to_stl_bytes(st.session_state.current_mesh)

        stl_filename = f"{st.session_state.source_name}_Soniprint.stl"

        st.download_button(
            label="📥 Download STL (for 3D printer)",
            data=stl_bytes,
            file_name=stl_filename,
            mime="application/octet-stream"
        )
        st.caption(f"File: `{stl_filename}`")

        st.markdown("""
        **Next steps:**
        1. Download the STL file above
        2. Open in **Cura** or **PrusaSlicer**
        3. Check for warnings (should be manifold)
        4. Slice and print!
        """)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
**About DataSoniPrint** — Convert any 2D dataset into a 3D-printable relief model.
Built with Streamlit, Plotly, trimesh, and ❤️
""")
