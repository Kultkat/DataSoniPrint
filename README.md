# DataSoniPrint — 2D/3D data → 3D‑printable tactile relief

Turn scientific data, math formulas, or images into a **watertight STL relief**
you can 3D‑print and *feel*. Built for **inclusive teaching**: a printed relief
lets students — especially those with visual impairments — grasp a graph, map, or
function by touch, not just by sight.

- **Live app:** https://datasoniprint.streamlit.app
- **Project home & example gallery:** https://www.physik.uzh.ch/~kaskoe/
- **Licence:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — free to
  use/share/adapt with credit to *“DataSoniPrint — B. Penning & K. König,
  University of Zurich.”*

A *“Tactile Data”* micro‑innovation project by
[Prof. Dr. Björn Penning](https://www.physik.uzh.ch/en/groups/penning.html) and
[Dipl.‑Des. M.Mus. Kaspar König](https://www.kasparkoenig.com), funded by
[ULF](https://www.ulf.uzh.ch) (Universitäre Lehrförderung), University of Zurich.

---

## What it does

Three ways to make a relief, all ending in a printable STL:

1. **📊 From a data column** — CSV / HDF5 / NetCDF / GRIB / ASDF. Pick X, Y, Z
   columns; the data is *spread* onto a 2D height grid.
2. **🔢 From a math formula** — type `sin(x)*cos(y)`, or pick from a curated
   **equation gallery** (Riemann ζ, Schrödinger states, gravity well, Mandelbrot,
   …). A **complex mode** renders `f(z)` with `z = x + i·y` (magnitude / real /
   imaginary / phase).
3. **🖼️ From an image** — a plot/photo becomes a relief by brightness. Optional
   **engraved text** or **braille** axis labels (via OCR).

Plus: a **live 3D preview**, bed‑size/height controls, and a one‑click **binary
STL** download.

---

## Run it locally

```bash
pip install -r requirements.txt
# OCR labels (optional) also need the Tesseract binary:
#   Ubuntu/Debian: sudo apt-get install tesseract-ocr
#   macOS:         brew install tesseract
streamlit run app.py
```

Python 3.10+ recommended.

---

## Under the hood

**The core idea:** an STL relief is a *heightfield* — one real height `z` over a
2D `(x, y)` grid. So every input mode is just a different way to produce a
normalized `z(x, y)` grid, which is then turned into a solid, watertight mesh.

The code is split in two:

| File | Role |
|------|------|
| [`app.py`](app.py) | Streamlit UI — widgets, session state, preview, download. No heavy math. |
| [`core_engine.py`](core_engine.py) | Pure, testable processing pipeline. No Streamlit imports. |

### The pipeline

```
input ──► normalized height grid z(x,y) in [0,1] ──► scale to bed ──► watertight mesh ──► binary STL
```

**1. Ingest & normalize**
- `load_file()` dispatches by extension to CSV/HDF5/NetCDF/GRIB/ASDF loaders and
  returns `{column: np.array}`.
- `normalize_to_01()` rescales any array to `[0, 1]` (NaNs handled), so the
  height scale is controlled purely by the mm sliders later.

**2. Build the 2D height grid** (one of):
- **Data:** `select_columns_for_axes()` + `spread_to_grid()` project 1‑D column
  data onto a `resolution × resolution` grid (reshape or scatter‑interpolate).
- **Formula:** `evaluate_formula()` parses with SymPy and evaluates on a NumPy
  meshgrid. Complex functions go through `_evaluate_complex()` (mpmath, so
  `zeta`/`gamma` work) and are projected to a real height. `mandelbrot_field()`
  is a dedicated escape‑time generator. `EQUATION_GALLERY` + `evaluate_gallery_item()`
  hold the curated presets (formula, domain, height, pole‑clipping/log options).
- **Image:** `build_image_relief()` reads brightness into a height grid;
  `detect_text_regions()` (Tesseract OCR) finds labels; `apply_labels()` either
  engraves the glyphs (negative height) or embosses standard **braille** dots
  (`text_to_braille_cells()`), sized in real millimetres.

**3. Scale to the print bed & mesh** — `scale_to_bed()`
- Fits the grid onto a ≤200 mm bed, adds a 1 mm base, and builds the top surface,
  bottom plate, and all four side walls so the mesh is **watertight**.
- Fully **vectorized** with NumPy (a pure‑Python loop took minutes on a 2 M‑face
  relief). Outward normals are set analytically per face group (top +Z, bottom
  −Z, walls ±X/±Y) instead of the slow adjacency‑graph `fix_normals()`.

**4. Export** — `mesh_to_stl_bytes()` writes a **binary** STL (≈5× smaller and
faster than ASCII).

### Design decisions that matter

- **Memory safety (images):** source images are capped to `MAX_RELIEF_DIM = 800`
  px on the longest side in `_load_grayscale()`. A 5 MB JPG can be 12+ megapixels
  → without a cap it becomes tens of millions of mesh faces and OOM‑kills the
  worker. 800 px keeps the worst case ~2 M faces while giving ~8 px/mm of detail.
- **True‑aspect bed:** `_bed_from_domain()` (in `app.py`) sizes the bed to the
  data/formula domain so a non‑square relief isn't stretched into a square.
- **Preview vs. print:** the on‑screen 3D preview exaggerates the Z axis for
  legibility (thin reliefs would otherwise look flat); the exported STL always
  uses the true millimetres shown on the sliders.
- **Memory readout:** `memory_usage_mb()` powers the sidebar gauge. A *hard* OOM
  kill (SIGKILL) can't be reported by the process; the gauge lets you watch the
  trend, and Python‑level `MemoryError`s are caught with a clear message.
- **Upload/limits:** `.streamlit/config.toml` sets `maxUploadSize`/`maxMessageSize`.
  Note the hosted tier has a fixed RAM budget — a very large *data* file can still
  exceed memory while being parsed, even though the upload itself is allowed.

---

## Project layout

```
app.py                     Streamlit UI (thin) + PROJECT_HOME/SOURCE links
core_engine.py             All processing (load, grid, formula, image, mesh, STL)
requirements.txt           Python deps
packages.txt               apt packages for Streamlit Cloud (tesseract-ocr)
.streamlit/config.toml     Upload/message size limits
site/                      Landing page (index.html) + example photos → UZH ~kaskoe
sample_data.csv            Example dataset
```

## Deployment

- **App:** Streamlit Community Cloud, repo `Kultkat/DataSoniPrint`, branch
  `streamlit-rebuild`, main file `app.py`. Auto‑redeploys on push. `packages.txt`
  installs Tesseract so OCR labels work.
- **Landing page:** `site/index.html` (+ the example JPGs) is uploaded to the UZH
  personal web space served at **www.physik.uzh.ch/~kaskoe**.

---

## Credits & licence

Developed at the University of Zurich (Physik‑Institut) as the *“Tactile Data”*
project by **Prof. Dr. Björn Penning** and **Dipl.‑Des. M.Mus. Kaspar König**,
funded by **ULF**. Released under **CC BY 4.0** — please keep the attribution
described above. See [`LICENSE`](LICENSE).
