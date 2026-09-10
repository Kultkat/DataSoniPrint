# DataSoniPrint

**Inclusive scientific data → 3D‑printable tactile relief**

*A “Tactile Data” micro‑innovation project by Prof. Dr. Björn Penning and
Dipl.‑Des. M.Mus. Kaspar König, University of Zurich, funded by ULF.*

---

## Aim

Scientific results are overwhelmingly presented as visual graphs and charts —
formats that exclude people with visual impairments and offer nothing tactile.
**DataSoniPrint** turns 2D/3D data, math formulas, and images into a
**watertight STL relief** you can 3D‑print and *feel*, so that understanding a
graph, map, or function doesn't depend on sight alone.

It is built for **inclusive teaching**: a printed relief lets students —
especially those with visual impairments — grasp an idea by touch, and it helps
every other learner too. A blind or low‑vision learner can hold and feel the
print; a sighted learner can read the same chart; everyone works from the same
underlying data.

---

## What it does

Three ways to make a relief, all ending in a printable STL:

1. **📊 From a data column** — CSV / HDF5 / NetCDF / GRIB / ASDF. Pick X, Y, Z
   columns; the data is *spread* onto a 2D height grid.
2. **🔢 From a math formula** — type `sin(x)*cos(y)`, or pick from a curated
   **equation gallery** (Riemann ζ, Schrödinger states, gravity well, Mandelbrot,
   Ricker/“Mexican‑hat” wavelet, Gamma …). A **complex mode** renders `f(z)` with
   `z = x + i·y` (magnitude / real / imaginary / phase).
3. **🖼️ From an image** — a plot or photo becomes a relief by brightness.
   Optional **engraved text** or **braille** axis labels, detected via OCR.

Plus: an on‑demand **live 3D preview**, print‑bed size / height controls, and a
one‑click **binary STL** download.

The underlying idea is simple: an STL relief is a *heightfield* — one real
height `z` over a 2D `(x, y)` grid. Every input mode is just a different way to
produce a normalized `z(x, y)`, which is then turned into a solid, watertight
mesh scaled to the print bed. See the [README](README.md) for the full
“under the hood” walkthrough.

---

## Accessibility by design

The goal is that **no single sense is required** to experience the data:

| Output           | Modality  | Who benefits                                              |
|------------------|-----------|-----------------------------------------------------------|
| **3D STL model** | Tactile   | Blind / low‑vision users can *feel* the data as a 3D print |
| **Braille / engraved labels** | Tactile | Axis labels readable by touch |
| **Live visual preview** | Visual | Sighted users check the relief before printing |

**Sonification** (hearing the data as sound) is part of the project's long‑term
vision — the *“Soni”* in DataSoniPrint — and earlier prototypes explored it with
real LIGO gravitational‑wave data. It is **not part of the current shipped app**;
the audio path is paused while the tactile/print workflow is the focus, and may
return in a later version.

---

## Run it

```bash
pip install -r requirements.txt
# OCR labels (optional) also need the Tesseract binary:
#   Ubuntu/Debian: sudo apt-get install tesseract-ocr
#   macOS:         brew install tesseract
streamlit run app.py
```

Python 3.10+ recommended. The hosted app runs on Streamlit Community Cloud
(branch `streamlit-rebuild`, `packages.txt` installs Tesseract for OCR).

---

## Project structure

| File / dir | Role |
|------------|------|
| [`app.py`](app.py) | Streamlit UI — widgets, session state, preview, download, maintenance controls. |
| [`core_engine.py`](core_engine.py) | Pure processing pipeline (load → grid → formula/image → mesh → STL). No Streamlit imports. |
| [`site/`](site/) | Landing page (`index.html`) + example photos, served at UZH `~kaskoe`. |
| `src/` | Earlier standalone prototypes (incl. the LIGO audio sonifier) — historical, not the shipped app. |

---

## Credits & licence

Developed at the University of Zurich (Physik‑Institut) as the *“Tactile Data”*
project by **Prof. Dr. Björn Penning** and **Dipl.‑Des. M.Mus. Kaspar König**,
funded by **ULF** (Universitäre Lehrförderung). Released under **CC BY 4.0** —
free to use, share, and adapt with credit to *“DataSoniPrint — B. Penning &
K. König, University of Zurich.”* See [`LICENSE`](LICENSE).
