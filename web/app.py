"""
app.py — Flask web backend for DataSoniPrint.

Endpoints:
    GET  /              → main UI
    POST /upload        → upload data file, return column info
    POST /preview       → quick line chart PNG of a column
    POST /process       → run sonification pipeline, return download links
    GET  /download/<id>/<type> → download WAV / STL / settings JSON
"""

import io
import json
import os
import secrets
import time
from pathlib import Path

from flask import (Flask, request, jsonify, send_file,
                   render_template, session)

from processing import (load_file, column_stats, process_file,
                         generate_preview_png, supported_extension,
                         data_column_to_relief_stl, scale_to_bed,
                         image_to_relief_stl, formula_grid_to_relief_stl)

app = Flask(__name__,
            template_folder="templates",
            static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024 * 1024  # 1 GB max upload

# In-memory store for processed results (keyed by session job ID).
# In production, swap for Redis or file-based storage.
_results = {}

# Auto-cleanup: keep results for max 30 minutes
MAX_RESULT_AGE = 1800


def _cleanup_old_results():
    now = time.time()
    expired = [k for k, v in _results.items() if now - v["ts"] > MAX_RESULT_AGE]
    for k in expired:
        del _results[k]


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Upload data file, return column list + stats."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    if not supported_extension(f.filename):
        return jsonify({"error": "Unsupported file format. Accepted: CSV, HDF5, NetCDF, GRIB, ASDF"}), 400

    try:
        contents = f.read()
        headers, columns, row_count = load_file(contents, f.filename)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    stats = column_stats(columns, headers)

    job_id = secrets.token_hex(12)
    _results[job_id] = {
        "ts": time.time(),
        "file_bytes": contents,
        "filename": f.filename,
        "headers": headers,
        "columns_cache": columns,
    }
    _cleanup_old_results()

    session["job_id"] = job_id

    return jsonify({
        "job_id": job_id,
        "filename": f.filename,
        "row_count": row_count,
        "headers": headers,
        "stats": stats,
    })


@app.route("/preview", methods=["POST"])
def preview():
    """Generate a quick line chart preview of a column."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    job_id = data.get("job_id") or session.get("job_id")
    if not job_id or job_id not in _results:
        return jsonify({"error": "No file uploaded or session expired"}), 400

    job = _results[job_id]
    columns = job.get("columns_cache")
    hdrs = job.get("headers")

    if columns is None or hdrs is None:
        return jsonify({"error": "Data not found, please re-upload"}), 400

    column = data.get("column")
    try:
        png_bytes = generate_preview_png(columns, hdrs, selected_column=column)
    except Exception as e:
        return jsonify({"error": f"Preview failed: {e}"}), 500

    return send_file(
        io.BytesIO(png_bytes),
        mimetype="image/png",
        download_name="preview.png",
    )


@app.route("/upload-image", methods=["POST"])
def upload_image():
    """Upload an image file for grayscale relief conversion."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}:
        return jsonify({"error": "Unsupported image format. Accepted: PNG, JPG, GIF, BMP, WebP"}), 400

    try:
        image_bytes = f.read()
        # Validate image can be loaded (but don't store it, we'll process on demand)
        from PIL import Image
        Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    job_id = secrets.token_hex(12)
    _results[job_id] = {
        "ts": time.time(),
        "file_bytes": image_bytes,
        "filename": f.filename,
        "file_type": "image",
    }
    _cleanup_old_results()

    session["job_id"] = job_id

    return jsonify({
        "job_id": job_id,
        "filename": f.filename,
        "file_type": "image",
        "message": "Image uploaded. Brightness will map to height (0-50mm).",
    })


@app.route("/example")
def example():
    """Serve bundled example CSV files for demo purposes."""
    name = request.args.get("name", "sample")
    base_dir = Path(__file__).parent.parent
    if name == "gw":
        sample_path = base_dir / "web" / "static" / "examples" / "gw_chirp.csv"
        dl_name = "gw_chirp.csv"
    else:
        sample_path = base_dir / "sample_data.csv"
        dl_name = "sample_data.csv"
    if not sample_path.exists():
        return jsonify({"error": "Sample file not found"}), 404
    return send_file(str(sample_path), mimetype="text/csv",
                     as_attachment=False, download_name=dl_name)


@app.route("/image-relief", methods=["POST"])
def image_relief():
    """Generate a relief STL from an uploaded image (grayscale → height)."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    job_id = data.get("job_id") or session.get("job_id")
    if not job_id or job_id not in _results:
        return jsonify({"error": "No image uploaded or session expired"}), 400

    job = _results[job_id]
    image_bytes = job.get("file_bytes")
    if image_bytes is None:
        return jsonify({"error": "Image data not found; please re-upload"}), 400

    width_mm          = max(10.0, min(float(data.get("width_mm", 200.0)),          200.0))
    depth_mm          = max(10.0, min(float(data.get("depth_mm",  200.0)),         200.0))
    height_scale_mm   = max(1.0,  min(float(data.get("height_scale_mm", 100.0)),   200.0))
    base_thickness_mm = max(1.0,  min(float(data.get("base_thickness_mm", 3.0)),   20.0))
    invert_brightness = bool(data.get("invert_brightness", False))
    spread            = max(0.5, min(float(data.get("spread", 1.0)),               3.0))

    scaled_w, scaled_d, scale_factor = scale_to_bed(width_mm, depth_mm, height_scale_mm)

    try:
        stl_bytes = image_to_relief_stl(
            image_bytes,
            width_mm=scaled_w,
            depth_mm=scaled_d,
            height_scale_mm=height_scale_mm,
            base_thickness_mm=base_thickness_mm,
            invert_brightness=invert_brightness,
            spread=spread,
        )
    except Exception as e:
        return jsonify({"error": f"STL generation failed: {e}"}), 500

    job["relief_stl"] = stl_bytes
    job["relief_source"] = "image"
    job["relief_filename"] = Path(job.get("filename", "image")).stem
    job["ts"] = time.time()

    return jsonify({
        "job_id": job_id,
        "width_mm": round(scaled_w, 2),
        "depth_mm": round(scaled_d, 2),
        "height_scale_mm": height_scale_mm,
        "scale_factor": round(scale_factor, 4),
        "ready": True,
        "message": "Image relief generated (brightness → height: 0-200mm)",
    })


@app.route("/relief", methods=["POST"])
def relief():
    """Generate a 3D relief STL from one data column or formula grid."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    job_id = data.get("job_id") or session.get("job_id")
    if not job_id or job_id not in _results:
        return jsonify({"error": "No file uploaded or session expired"}), 400

    job = _results[job_id]

    # Check if this is formula data
    is_formula = job.get("file_type") == "formula"
    n_points = None  # Only used for data mode

    if is_formula:
        import numpy as np
        formula_data = job.get("formula_data")
        if formula_data is None:
            return jsonify({"error": "Formula data not available"}), 400

        # Formula data is 2D grid — use dedicated converter with default 100mm cube (1:1:1)
        # Square formula grids map to square physical space → radial formulas print round
        width_mm          = max(10.0, min(float(data.get("width_mm", 100.0)),          200.0))
        depth_mm          = max(10.0, min(float(data.get("depth_mm",  100.0)),         200.0))
        height_scale_mm   = max(10.0, min(float(data.get("height_scale_mm", 100.0)),   200.0))
        base_thickness_mm = max(1.0,  min(float(data.get("base_thickness_mm", 3.0)),   20.0))
        spread            = max(0.5, min(float(data.get("spread", 1.0)),               3.0))
        invert_volume     = bool(data.get("invert_volume", False))

        # Apply transformations to formula grid
        grid_transformed = formula_data.copy()

        # Apply spread function (power law for contrast)
        if spread != 1.0:
            grid_transformed = np.power(np.clip(grid_transformed, 0, 1), 1.0 / spread)

        # Invert volume if requested
        if invert_volume:
            grid_transformed = 1.0 - grid_transformed

        scaled_w, scaled_d, scale_factor = scale_to_bed(width_mm, depth_mm, height_scale_mm)

        try:
            stl_bytes = formula_grid_to_relief_stl(
                grid_transformed,
                width_mm=scaled_w,
                depth_mm=scaled_d,
                height_scale_mm=height_scale_mm,
                base_thickness_mm=base_thickness_mm,
            )
        except Exception as e:
            return jsonify({"error": f"STL generation failed: {e}"}), 500
    else:
        # Standard column-based data
        columns = job.get("columns_cache")
        if columns is None:
            return jsonify({"error": "Column data not available; please re-upload"}), 400

        column = data.get("column")
        if not column or column not in columns:
            column = (job.get("headers") or [None])[0]
        if not column or column not in columns:
            return jsonify({"error": "No valid column specified"}), 400
        column_data = columns[column]

        width_mm          = max(10.0, min(float(data.get("width_mm", 180.0)),          200.0))
        depth_mm          = max(5.0,  min(float(data.get("depth_mm",  20.0)),          200.0))
        height_scale_mm   = max(1.0,  min(float(data.get("height_scale_mm",  20.0)),    80.0))
        base_thickness_mm = max(1.0,  min(float(data.get("base_thickness_mm", 3.0)),    20.0))
        n_points          = max(10,   min(int(data.get("n_points", 200)),              500))

        scaled_w, scaled_d, scale_factor = scale_to_bed(width_mm, depth_mm, height_scale_mm)

        try:
            stl_bytes = data_column_to_relief_stl(
                column_data,
                width_mm=scaled_w,
                depth_mm=scaled_d,
                height_scale_mm=height_scale_mm,
                base_thickness_mm=base_thickness_mm,
                n_points=n_points,
            )
        except Exception as e:
            return jsonify({"error": f"STL generation failed: {e}"}), 500

    job["relief_stl"] = stl_bytes
    if is_formula:
        job["relief_filename"] = "formula"
    else:
        job["relief_column"] = column
    job["ts"] = time.time()

    response = {
        "job_id": job_id,
        "column": job.get("relief_column", "formula"),
        "width_mm": round(scaled_w, 2),
        "depth_mm": round(scaled_d, 2),
        "height_scale_mm": height_scale_mm,
        "scale_factor": round(scale_factor, 4),
        "ready": True,
    }
    if n_points is not None:
        response["n_points"] = n_points

    return jsonify(response)


@app.route("/process", methods=["POST"])
def process():
    """Run the sonification pipeline with user parameters."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    job_id = data.get("job_id") or session.get("job_id")
    if not job_id or job_id not in _results:
        return jsonify({"error": "No file uploaded or session expired"}), 400

    job = _results[job_id]
    file_bytes = job.get("file_bytes")
    if file_bytes is None:
        return jsonify({"error": "File data not found, please re-upload"}), 400

    params = {
        "columns": data.get("columns", []),
        "column": data.get("column"),
        "spread": data.get("spread", 0.35),
        "speed": data.get("speed", 0.5),
        "volume": data.get("volume", 0.7),
    }

    try:
        result = process_file(file_bytes, job["filename"], params)
    except Exception as e:
        return jsonify({"error": f"Processing failed: {e}"}), 500

    # Free cached columns to save memory now that we have outputs
    job.pop("columns_cache", None)

    job["wav"] = result["wav"]
    job["stl"] = result["stl"]
    job["settings"] = result["settings"]
    job["ts"] = time.time()

    return jsonify({
        "job_id": job_id,
        "settings": result["settings"],
        "headers": result["headers"],
        "ready": True,
    })


@app.route("/fetch-formula-html", methods=["POST"])
def fetch_formula_html():
    """Fetch HTML from URL and extract formula text using heuristics."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        import requests
        from bs4 import BeautifulSoup
        import re

        # Fetch page
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text and look for formula patterns
        text = soup.get_text(separator=' ')

        # Try to extract formulas: look for V(, f(, equations with =, etc.
        # For Wikipedia, look in math notation
        formulas = []

        # Pattern 1: V(x,y) = ... or f(x,y) = ...
        pattern1 = re.findall(r'[Vf]\([^)]*\)\s*=\s*[^\n.;]{10,100}', text)
        formulas.extend(pattern1)

        # Pattern 2: Look for math tags content
        math_tags = soup.find_all(['math', 'script', 'span'])
        for tag in math_tags:
            if 'class' in tag.attrs and 'mwe-math' in str(tag.attrs.get('class', '')):
                if tag.get_text(strip=True):
                    formulas.append(tag.get_text(strip=True))

        # Return first found formula or summary
        formula_text = formulas[0] if formulas else "Could not extract formula"

        return jsonify({
            "formula_text": formula_text,
            "url": url,
        })
    except ImportError:
        return jsonify({
            "error": "requests/BeautifulSoup not installed. Paste formula manually.",
            "formula_text": None
        }), 400
    except Exception as e:
        return jsonify({"error": f"Failed to fetch: {str(e)}"}), 400


@app.route("/evaluate-formula", methods=["POST"])
def evaluate_formula_route():
    """Evaluate formula and create data grid for relief generation."""
    from processing import evaluate_formula

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    formula = data.get("formula", "").strip()
    resolution = max(10, min(int(data.get("resolution", 100)), 300))

    if not formula:
        return jsonify({"error": "No formula provided"}), 400

    try:
        result = evaluate_formula(formula, resolution)

        # Store formula grid as a temporary dataset
        job_id = secrets.token_hex(12)
        import numpy as np
        grid_data = result['data']

        _results[job_id] = {
            "ts": time.time(),
            "formula": formula,
            "formula_data": grid_data,
            "filename": f"formula_{job_id[:8]}.npy",
            "file_type": "formula",
        }
        _cleanup_old_results()

        return jsonify({
            "job_id": job_id,
            "formula": formula,
            "resolution": resolution,
            "z_range": [result['z_min'], result['z_max']],
            "message": f"Formula evaluated at {resolution}×{resolution} resolution",
        })
    except Exception as e:
        return jsonify({"error": f"Formula evaluation failed: {str(e)}"}), 400


@app.route("/download/<job_id>/<file_type>")
def download(job_id, file_type):
    """Download a processed output file."""
    if job_id not in _results:
        return jsonify({"error": "Job not found or expired"}), 404

    job = _results[job_id]
    base = os.path.splitext(job.get("filename", "data"))[0]

    if file_type == "wav":
        data = job.get("wav")
        if not data:
            return jsonify({"error": "WAV not ready"}), 404
        return send_file(
            io.BytesIO(data),
            mimetype="audio/wav",
            as_attachment=True,
            download_name=f"{base}_sonified.wav",
        )
    elif file_type == "stl":
        data = job.get("stl")
        if not data:
            return jsonify({"error": "STL not ready"}), 404
        return send_file(
            io.BytesIO(data),
            mimetype="application/sla",
            as_attachment=True,
            download_name=f"{base}_terrain.stl",
        )
    elif file_type == "settings":
        settings = job.get("settings")
        if not settings:
            return jsonify({"error": "Settings not ready"}), 404
        return send_file(
            io.BytesIO(json.dumps(settings, indent=2).encode()),
            mimetype="application/json",
            as_attachment=True,
            download_name=f"{base}_settings.json",
        )
    elif file_type == "relief":
        data = job.get("relief_stl")
        if not data:
            return jsonify({"error": "Relief STL not ready; call /relief first"}), 404
        relief_name = job.get("relief_filename") or job.get("relief_column", "data")
        return send_file(
            io.BytesIO(data),
            mimetype="application/sla",
            as_attachment=True,
            download_name=f"{relief_name}_relief.stl",
        )
    else:
        return jsonify({"error": f"Unknown file type: {file_type}"}), 400


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("DataSoniPrint Web — http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
