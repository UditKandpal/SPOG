# SPOG GPU — Geometry-Driven GPU Rendering for High-Fidelity UAV Solar Farm Orthomosaics

**SPOG GPU** is a post-processing renderer that takes a textured 3-D mesh produced by any photogrammetry pipeline (WebODM, OpenDroneMap, Agisoft Metashape) and generates a high-fidelity 16-bit orthomosaic specifically optimised for solar PV farm inspection.

The rendering stage runs entirely on the GPU via OpenGL 3.3 / ModernGL + EGL (headless, no display server), achieving ~100× speedup over CPU-based alternatives while preserving every algorithmic guarantee from the original SPOG paper.

> **Research context:** This implementation accompanies the paper  
> *"SPOG GPU: Geometry-Driven GPU Rendering for High-Fidelity UAV Solar Farm Orthomosaics"*  
> submitted to IEEE Geoscience and Remote Sensing Letters (IEEE GRSL).

---

## Why SPOG GPU?

Standard photogrammetry outputs suffer from rendering artefacts that corrupt solar panel texture: Z-fighting on coplanar panel faces, UV streaking at tile boundaries, radiometric seams between flight-path overpasses, and blurred defect signatures caused by multi-band blending. These are not SfM/MVS errors — they are rendering-stage failures that occur even when the 3-D geometry is correct.

SPOG GPU resolves all five artefact classes through domain-specific rendering primitives designed for flat, repeating solar panel geometry:

| Primitive | What it fixes |
|---|---|
| Explicit UV wrapping `fract(u,v)` | Texture streaking at tiled panel boundaries |
| Nadir-confidence weighting `\|n_z\|` | Seam colour discontinuities at overlapping tiles |
| Slope-dependent depth bias `glPolygonOffset(1,1)` | Z-fighting on coplanar panel faces |
| One-directional radiometric normalisation | Brightness seams between UAV overpasses |
| Geometry-driven ROI mask + EDT gap fill | Background contamination and mesh-hole artefacts |

The result is a 16-bit uncompressed TIFF orthomosaic where cell boundaries, busbar edges, and micro-textural defect signatures are faithfully preserved — suitable for direct input to both automated defect detectors and GIS workflows.

---

## Algorithm

SPOG GPU implements five novel rendering primitives on top of a standard orthographic GPU pipeline:

### 1. Explicit UV Wrapping (Equation 1)

```
u' = fract(u) = u − ⌊u⌋,    v' = fract(v)
```

Solar panel textures tile repeatedly across the array. UV coordinates outside `[0, 1]` occur naturally; standard clamping produces edge streaks. `fract()` wraps them explicitly in the GLSL fragment shader.

### 2. Nadir-Alignment Confidence Weighting

```
I_out(x,y) = Σ_f |n_z(f)| · c(f) / Σ_f |n_z(f)|
```

Each fragment is weighted by the absolute nadir component of its face normal. Faces tilted away from vertical contribute less to the accumulated colour, favouring the most nadir-aligned view for each ground point. Implemented via an additive float32 FBO (`GL_ONE + GL_ONE`) with per-fragment normalisation on readback.

### 3. Slope-Dependent Depth Bias (Equation 3)

```
glPolygonOffset(factor=1.0, units=1.0)
```

Hardware equivalent of the paper's `d ≥ d_buffer − δ` epsilon test. Eliminates Z-fighting on geometrically coplanar panels without a software Z-buffer.

### 4. One-Directional Radiometric Normalisation

```
μ_ref = median({μ_tile})

if μ_tile < μ_ref:
    L_norm = clip(L_tile + 0.5 × (μ_ref − μ_tile),  0, 255)
else:
    L_norm = L_tile    ← bright tiles unchanged
```

Operates on the CIE-LAB L channel only (chrominance preserved). Under-exposed tiles are brightened toward the scene median; over-exposed tiles with white busbars and frame edges are left untouched. Eliminates radiometric seams without spatial blending.

### 5. Defect-Aware Unsharp Masking (Equation 4)

```
I_sharp = I + α · max(0, |I − I_blur| − τ) · sign(I − I_blur)
α = 0.3,   τ = 3 DN,   σ = 0.8 px
```

Threshold-gated unsharp masking. Only edge responses above `τ = 3` DN are amplified — genuine defect signatures (micro-cracks ~10–30 DN, hotspot boundaries ~15–50 DN). Noise-level fluctuations (< 3 DN) are suppressed.

### Panel ROI Hard-Boundary Contouring

After GPU rasterisation, the rendered mesh footprint is captured as a binary mask, morphologically closed to bridge inter-row gaps, and applied as a hard binary cutoff. Background (vegetation, roads, sky) becomes pure black with a pixel-precise boundary. No colour segmentation, no manual annotation.

### Distance-Aware Black-Pixel Fill (EDT)

Mesh holes and seam gaps are filled using `scipy.ndimage.distance_transform_edt` — every black pixel receives the colour of its nearest rendered neighbour. No Gaussian blur; no blending artefacts.

---

## Visual Results

Side-by-side orthomosaic comparisons (SPOG GPU vs WebODM vs Agisoft Metashape) with zoomed panel-level detail:

| Dataset | Comparison PDF |
|---|---|
| 200-image (459 × 252 m) | [📄 200_comparison.pdf](comparisons/200_comparison.pdf) |
| 500-image (1,176 × 389 m) | [📄 500_comparison.pdf](comparisons/500_comparison.pdf) |

Each PDF shows the full orthomosaic side-by-side and zoomed crop regions highlighting how SPOG GPU preserves cell boundaries, busbar edges, and defect signatures that are blurred or artefact-corrupted in WebODM and Agisoft outputs.

---

## Results

**200-image dataset** (459 × 252 m scene, 339,627-face mesh):

| Metric | SPOG GPU | WebODM | Agisoft |
|---|---|---|---|
| Resolution | **147.2 MP** | 43.8 MP | 255.8 MP |
| Bit depth | **16-bit** | 8-bit | 8-bit |
| GPU render time | **9.4 s** | — | — |
| Total wall time | **142 s** | — | — |
| CNN Average Precision ↑ | **0.9505** | 0.9274 | 0.9487 |
| CNN False Positives ↓ | **7** | 19 | 20 |
| Detection Reliability Score ↑ | **0.9389** | 0.9151 | 0.9393 |
| Cross-domain AP (SPOG→Agisoft) | **0.9433** | — | — |

**500-image dataset** (1,176 × 389 m scene, 351,806-face mesh):

| Metric | SPOG GPU | WebODM |
|---|---|---|
| Resolution | **190.7 MP** | 177.4 MP |
| Bit depth | **16-bit** | 8-bit |
| GPU render time | **13.0 s** | — |
| Total wall time | **176 s** | — |

---

## File Structure

```
.
├── spog_gpu.py          # Core GPU renderer — SPOGGPUGenerator class
├── spog_gpu_roi.py      # Panel ROI hard-boundary contouring — PanelROIMasker
├── spog_gpu_runner.py   # CLI entry point
├── ortho_metrics_test.py # Standalone comparison tool (compare any two TIFFs)
├── model_1/             # Example input: OBJ mesh + PNG textures
│   ├── drm_textured_model_geo.obj
│   ├── drm_textured_model_geo.mtl
│   └── drm_textured_model_geo_material*.png
└── results/             # Generated outputs (created on first run)
    └── model_1/
        ├── spog_gpu_orthomosaic.tif
        ├── spog_gpu_orthomosaic_masked.tif
        ├── spog_gpu_orthomosaic_preview.png
        └── spog_gpu_orthomosaic_metadata.json
```

---

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

GPU requirements: NVIDIA GPU with EGL support (tested on RTX A4000). The EGL backend runs headless — no display server, Xvfb, or VirtualGL required.

### Run on the bundled example

```bash
python3 spog_gpu_runner.py --model model_1
```

### Run on your own model

```bash
# Auto-detects any .obj file in the folder
python3 spog_gpu_runner.py --model /path/to/your/model --size 16384

# Explicit OBJ path (use when the folder has multiple .obj files)
python3 spog_gpu_runner.py --model /path/to/your/model --obj /path/to/your/model/mesh.obj
```

The model folder must contain the mesh `.obj` file and its associated texture images (PNG, JPG, or TIF). If the OBJ is named `drm_textured_model_geo.obj` (WebODM / OpenDroneMap default) it is auto-detected; otherwise any `.obj` in the folder is used, or you can pass `--obj` explicitly.

### Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `model_1` | Path to model folder (OBJ + textures) |
| `--obj` | auto | Explicit path to `.obj` file |
| `--size` | `16384` | Maximum output side in pixels |
| `--no-roi` | off | Skip panel ROI masking |
| `--debug-roi` | off | Save intermediate ROI mask layers |

---

## Python API

```python
from spog_gpu import SPOGGPUGenerator

gen = SPOGGPUGenerator(debug=True)
ok, msg = gen.generate(
    obj_file      = "model_1/drm_textured_model_geo.obj",
    texture_files = ["model_1/drm_textured_model_geo_material0000_map_Kd.png", ...],
    output_path   = "results/model_1/spog_gpu_orthomosaic.tif",
    image_size    = 16384,
    apply_roi_mask = True,
)
print(msg)  # prints timing, resolution, and quality metrics
```

**SPOGGPUGenerator.generate()** returns `(success: bool, message: str)`.  
Metrics are also available on `gen.pattern_metrics` after the call.

---

## Comparing Two Orthomosaics

```bash
python3 ortho_metrics_test.py \
    --images results/model_1/spog_gpu_orthomosaic.tif webodm_output.tif \
    --names "SPOG GPU" "WebODM" \
    --output comparison_report.txt
```

Computed metrics: Edge Density, High-Frequency Energy, Detected Grid Lines, Texture Uniformity, Edge Orientation Regularity, Global Contrast, Panel-like Structures, Pattern Regularity Score.

---

## Citation

If you use SPOG GPU in research, please cite:

```bibtex
@article{kandpal2025spoggpu,
  title   = {{SPOG GPU}: Geometry-Driven {GPU} Rendering for High-Fidelity
             {UAV} Solar Farm Orthomosaics},
  author  = {Kandpal, Udit and Das, Debasis},
  journal = {IEEE Geoscience and Remote Sensing Letters},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

This project is provided for research and commercial use.

**Author:** Udit Kandpal, IIT Jodhpur  
**Contact:** m24cse027@iitj.ac.in
