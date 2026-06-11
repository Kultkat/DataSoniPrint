"""
DataSoniPrint Streamlit App — Phase 1-4 Implementation

A clean, Python-native interface for converting 2D data into 3D-printable relief STLs.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import io
import tempfile

from core_engine import (
    load_file, normalize_to_01, select_columns_for_axes,
    evaluate_formula, generate_preview_mesh, scale_to_bed, mesh_to_stl_bytes,
    image_to_relief_array, spread_to_grid
)

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


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: Data Ingestion UI
# ─────────────────────────────────────────────────────────────────────────────

st.header("📊 Phase 1: Load Your Data")

# File upload
uploaded_file = st.file_uploader(
    "Upload a data file (CSV, HDF5, NetCDF, GRIB, ASDF):",
    type=["csv", "h5", "hdf5", "hdf", "nc", "nc4", "netcdf", "grib", "grib2", "grb", "grb2", "asdf"]
)

if uploaded_file:
    try:
        file_bytes = uploaded_file.read()
        headers, columns = load_file(file_bytes, uploaded_file.name)

        st.session_state.loaded_file = uploaded_file.name
        st.session_state.headers = headers
        st.session_state.columns = columns

        st.success(f"✓ Loaded {uploaded_file.name} — {len(headers)} columns, {len(columns[headers[0]])} rows")

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

    except Exception as e:
        st.error(f"❌ Error loading file: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 & 3: Mode Selection (Data vs Formula vs Image)
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.headers:
    st.header("🔄 Select Mode")

    mode = st.radio(
        "How would you like to generate the relief?",
        ["📊 From Data Column", "🔢 From Math Formula", "🖼️ From Image"]
    )

    # ─────────────────────────────────────────────────────────────────────────
    # MODE 1: Data Column
    # ─────────────────────────────────────────────────────────────────────────

    if mode == "📊 From Data Column":
        st.subheader("Select Columns for X, Y, Z Axes")

        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("X-axis (width):", st.session_state.headers, key="x_col")
        with col2:
            y_col = st.selectbox("Y-axis (depth):", st.session_state.headers, index=min(1, len(st.session_state.headers)-1), key="y_col")
        with col3:
            z_col = st.selectbox("Z-axis (height):", st.session_state.headers, index=min(2, len(st.session_state.headers)-1), key="z_col")

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
                    st.success(
                        f"✓ Spread {len(z_data)} Z samples → {z_grid.shape[0]}×{z_grid.shape[1]} relief grid"
                    )
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # MODE 2: Formula
    # ─────────────────────────────────────────────────────────────────────────

    elif mode == "🔢 From Math Formula":
        st.subheader("Enter a Mathematical Formula")

        formula = st.text_input(
            "Formula (e.g., 'sin(x)*cos(y)', 'exp(-x**2 - y**2)', 'sqrt(x**2 + y**2)'):",
            value="sin(x)*cos(y)"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            x_range_min = st.number_input("X range (min):", value=-5.0)
            x_range_max = st.number_input("X range (max):", value=5.0)
        with col2:
            y_range_min = st.number_input("Y range (min):", value=-5.0)
            y_range_max = st.number_input("Y range (max):", value=5.0)
        with col3:
            resolution = st.slider("Grid Resolution:", 20, 200, 80, step=10)

        if st.button("▶ Generate from Formula", key="gen_formula"):
            with st.spinner("Evaluating formula and generating mesh..."):
                try:
                    z_grid = evaluate_formula(
                        formula,
                        x_range=(x_range_min, x_range_max),
                        y_range=(y_range_min, y_range_max),
                        resolution=resolution
                    )
                    st.session_state.current_z_grid = z_grid
                    st.success("✓ Formula evaluated and mesh generated")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # MODE 3: Image
    # ─────────────────────────────────────────────────────────────────────────

    elif mode == "🖼️ From Image":
        st.subheader("Upload a Grayscale Image")

        image_file = st.file_uploader("Upload PNG or JPG:", type=["png", "jpg", "jpeg"])

        if image_file:
            col1, col2 = st.columns(2)
            with col1:
                height_min = st.slider("Minimum height (mm):", 0.5, 5.0, 0.5, step=0.1)
            with col2:
                height_max = st.slider("Maximum height (mm):", 0.5, 10.0, 5.0, step=0.1)

            if st.button("▶ Generate from Image", key="gen_image"):
                with st.spinner("Converting image to relief..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                            tmp.write(image_file.getbuffer())
                            tmp_path = tmp.name

                        z_grid = image_to_relief_array(tmp_path, height_min, height_max)
                        st.session_state.current_z_grid = z_grid
                        st.success("✓ Image converted to relief")

                        Path(tmp_path).unlink()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: STL Generation & Scaling
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.current_z_grid is not None:
    st.header("⚙️ Phase 4: Configure & Export")

    col1, col2, col3 = st.columns(3)
    with col1:
        width_mm = st.slider("Width (mm):", 50, 200, 150, step=10)
    with col2:
        depth_mm = st.slider("Depth (mm):", 10, 200, 100, step=10)
    with col3:
        height_mm = st.slider("Height scale (mm):", 5, 40, 20, step=1)

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📐 Current bounding box: {width_mm}×{depth_mm}×{height_mm}mm (max 200mm per side)")
    with col2:
        smooth = st.checkbox("Apply smoothing (Gaussian blur) for easier printing")

    # ─────────────────────────────────────────────────────────────────────────
    # Live Preview (low-res)
    # ─────────────────────────────────────────────────────────────────────────

    st.subheader("📱 Live 3D Preview (low resolution)")

    preview_grid = generate_preview_mesh(st.session_state.current_z_grid)

    # Create Plotly surface plot
    h, w = preview_grid.shape
    x = np.linspace(0, width_mm, w)
    y = np.linspace(0, depth_mm, h)
    z = preview_grid * height_mm + 1.0  # 1mm base

    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale="Viridis")])
    fig.update_layout(
        title="3D Relief Preview",
        scene=dict(
            xaxis_title="Width (mm)",
            yaxis_title="Depth (mm)",
            zaxis_title="Height (mm)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=500,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Generate High-Res STL
    # ─────────────────────────────────────────────────────────────────────────

    if st.button("✨ Generate High-Res STL", key="gen_stl"):
        with st.spinner("Creating high-resolution mesh for 3D printing..."):
            try:
                mesh, dims = scale_to_bed(
                    st.session_state.current_z_grid,
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

        st.download_button(
            label="📥 Download STL (for 3D printer)",
            data=stl_bytes,
            file_name="datasoniprint_relief.stl",
            mime="application/octet-stream"
        )

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
