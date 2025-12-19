# Solar Panel Orthomosaic Generator

An algorithm for generating high-quality orthomosaic images from 3D models, specifically optimized for solar panel installations. This tool processes OBJ files with textures to create top-down orthographic projections with advanced pattern preservation and recognition metrics.

## 🌟 Features

- **Solar Panel Optimized**: Custom face filtering and pattern enhancement specifically designed for solar panel arrays
- **Batch Processing**: Process multiple OBJ files automatically with organized output structure
- **Pattern Recognition Metrics**: Comprehensive quantitative analysis including edge density, texture uniformity, and line detection
- **High-Quality Rendering**: Mipmapped texture filtering, bilinear interpolation, and adaptive LOD selection
- **EXIF Integration**: Extracts GPS, altitude, and camera calibration data when available
- **Performance Optimized**: Numba-accelerated computations and spatial indexing for fast rendering
- **Quality Comparison**: Built-in metrics calculator to compare different orthomosaic generation methods

## 📁 Project Structure

```
.
├── main_orthomosaic.py          # Main entry point for batch processing
├── solar_panel_orthomosaic.py   # Core algorithm and rendering engine
├── ortho_metrics_test.py        # Metrics comparison tool
├── model_1/                     # Example input data
│   ├── *.obj                    # 3D model geometry
│   ├── *.mtl                    # Material definitions
│   └── *.jpg/*.png              # Texture images
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy pillow opencv-python scipy scikit-image scikit-learn numba
```

### Basic Usage

#### 1. Process a Single OBJ File

```bash
python main_orthomosaic.py --input model_1/model.obj --output ./results
```

#### 2. Batch Process All OBJ Files in a Directory

```bash
python main_orthomosaic.py --input ./models --output ./orthomosaics
```

#### 3. Custom Output Size with Force Reprocess

```bash
python main_orthomosaic.py --input ./models --output ./results --size 8192 --force
```

## 📖 Detailed Usage

### `main_orthomosaic.py` - Main Processing Script

This is your primary interface for generating orthomosaics from OBJ files.

**Command-Line Arguments:**

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--input` | `-i` | Input OBJ file or directory | Required |
| `--output` | `-o` | Output directory for orthomosaics | Required |
| `--size` | `-s` | Output image size (longest side) | 6144 |
| `--force` | `-f` | Force reprocessing existing outputs | False |
| `--max-files` | `-m` | Maximum number of files to process | Unlimited |
| `--debug` | `-d` | Enable debug output | False |

**Examples:**

```bash
# Process current directory
python main_orthomosaic.py --input . --output ./orthomosaics

# Process with 4K output
python main_orthomosaic.py --input model_1/ --output ./output_4k --size 4096

# Process first 5 files only (useful for testing)
python main_orthomosaic.py --input ./models --output ./test --max-files 5 --debug
```

**Output Structure:**

The script creates an organized output directory:

```
output/
├── model_1/
│   ├── model_1_orthomosaic.png       # Main output image
│   ├── model_1_orthomosaic.jpg       # JPEG version (95% quality)
│   └── model_1_metadata.json         # Processing metadata
├── processing_report.json             # Detailed JSON report
└── processing_summary.txt             # Human-readable summary
```

### `solar_panel_orthomosaic.py` - Core Algorithm

This file contains the main orthomosaic generation logic with the `SolarPanelICPRGenerator` class.

**Key Features:**

1. **Enhanced OBJ Parsing**: Handles vertices, texture coordinates, and face groups
2. **Material & Texture Loading**: Parses MTL files and loads associated textures
3. **Mipmap Generation**: Creates multiple LOD levels for better texture quality
4. **Optimized Rasterization**: Fast triangle rasterization with depth buffering
5. **Pattern Metrics**: Calculates 8+ metrics for pattern recognition evaluation
6. **Solar Panel Enhancement**: Post-processing optimized for panel visibility

**Direct Usage in Python:**

```python
from solar_panel_orthomosaic import generate_enhanced_orthomosaic

success, result = generate_enhanced_orthomosaic(
    obj_file="model_1/model.obj",
    texture_files=["model_1/texture1.jpg", "model_1/texture2.jpg"],
    output_path="output/result.png",
    image_files=["model_1/images/img001.jpg"],  # Optional: for EXIF
    image_size=6144
)

if success:
    print(result)  # Prints ICPR metrics summary
else:
    print(f"Error: {result}")
```

**Algorithm Pipeline:**

```
1. Load OBJ/MTL files → Parse geometry and materials
2. Load textures → Generate mipmaps for each texture
3. Calculate bounds → Determine world-to-pixel transformation
4. Build spatial index → Accelerate rendering with KDTree
5. Rasterize triangles → Render with depth buffering
6. Calculate metrics → Edge density, texture uniformity, etc.
7. Post-process → Adaptive sharpening and contrast enhancement
8. Save output → PNG, JPG, and metadata JSON
```

### `ortho_metrics_test.py` - Metrics Comparison Tool

Compare multiple orthomosaic images using quantitative pattern recognition metrics.

**Command-Line Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--images` | Paths to 2+ orthomosaic images | Required |
| `--names` | Display names for each image | Required |
| `--output` | Output report path | `multi_comparison_report.txt` |
| `--method` | Alignment method: `resize` or `crop` | `resize` |

**Examples:**

```bash
# Compare 4 methods
python ortho_metrics_test.py \
    --images webodm.tif dronedeploy.tif metashape.tif custom.png \
    --names WebODM DroneDeploy Metashape "Our Method"

# Compare 2 methods with cropping
python ortho_metrics_test.py \
    --images method1.png method2.png \
    --names "Method A" "Method B" \
    --method crop \
    --output comparison_results.txt
```

**Output Metrics:**

The tool calculates 8 comprehensive metrics:

1. **Edge Density**: Ratio of edge pixels (higher = more structured)
2. **High-Frequency Energy**: FFT-based pattern regularity measure
3. **Detected Grid Lines**: Number of linear features (Hough transform)
4. **Texture Uniformity**: Local standard deviation (lower = less smoothed)
5. **Edge Orientation Regularity**: Consistency of edge directions (lower = better)
6. **Global Contrast**: Standard deviation of pixel intensities
7. **Panel-like Structures**: Count of rectangular contours
8. **Pattern Regularity Score**: Autocorrelation-based periodicity measure

**Sample Output:**

```
ORTHOMOSAIC QUANTITATIVE COMPARISON - ALL METHODS
==================================================================
Metric                                         WebODM  DroneDeploy  Metashape  Our Method
------------------------------------------------------------------
Edge Density (edges/px)                       0.045123    0.052341   0.048762   0.063421
  └─ Best: Our Method                             -28%        -17%       -23%       BEST
High-Frequency Energy (×10⁹)                  12.34       15.67      14.23      18.92
  └─ Best: Our Method                             -35%        -17%       -25%       BEST
...
```

## 🎯 Use Cases

### 1. Solar Panel Installation Analysis
Generate high-quality orthomosaics for:
- Panel alignment verification
- Installation quality assessment
- Defect detection preparation
- Documentation and reporting

### 2. Research and Development
- Compare different orthomosaic generation algorithms
- Benchmark pattern preservation capabilities
- Evaluate texture quality and detail retention
- Generate datasets for machine learning

### 3. Batch Processing Workflows
- Process entire directories of 3D models
- Standardized output for multiple sites
- Automated report generation
- Quality control across projects

## 🔬 Pattern Recognition Metrics Explained

### Edge Density
Measures the proportion of edge pixels in the image. Solar panel arrays should have high edge density due to panel boundaries and frame structures.

**Good values**: 0.05 - 0.15

### High-Frequency Energy
Analyzes the Fourier transform to detect regular patterns. Higher values indicate preserved panel grid structures.

**Good values**: 15+ (×10⁹ scale)

### Line Density
Counts detected straight lines using Hough transform. More lines indicate better preservation of panel boundaries.

**Good values**: 20+ lines per 10,000 pixels

### Texture Uniformity
Measures local texture variation. Lower values indicate less over-smoothing.

**Good values**: 0.1 - 0.3 (lower is better for preserving detail)

### Edge Regularity
Analyzes consistency of edge orientations. Lower values indicate more regular structures.

**Good values**: 0.3 - 0.6 (lower is better)

## ⚙️ Technical Details

### Algorithm Optimizations

1. **Numba JIT Compilation**: Critical functions compiled for 10-100x speedup
2. **Spatial Indexing**: KDTree acceleration for face queries
3. **Batch Processing**: Groups faces by material for cache efficiency
4. **Mipmap Filtering**: Reduces texture aliasing and improves quality
5. **Early Rejection**: Bounding box tests skip non-visible triangles

### Memory Management

The algorithm is designed to handle large models efficiently:
- Processes faces in batches (1000 at a time)
- Uses NumPy for memory-efficient arrays
- Explicit garbage collection between major steps
- Typical memory usage: 2-8GB for 6144px output

### Supported Formats

**Input:**
- OBJ files with triangulated meshes
- MTL material definitions
- PNG, JPG, JPEG, TGA, BMP textures
- Optional: JPEG/TIFF images with EXIF data

**Output:**
- PNG (lossless, optimized)
- JPEG (95% quality)
- JSON metadata with metrics

## 🐛 Troubleshooting

### Common Issues

**"No texture files found"**
- Ensure texture files are in the same directory as OBJ or in a `textures/` subdirectory
- Check that MTL file correctly references texture filenames
- Supported extensions: .png, .jpg, .jpeg, .tga, .bmp

**"Output already exists"**
- Use `--force` flag to reprocess existing outputs
- Or manually delete the output directory

**Memory errors with large models**
- Reduce `--size` parameter (try 4096 or 2048)
- Process fewer files at once using `--max-files`
- Close other applications to free RAM

**Blank or incomplete output**
- Check that OBJ file contains valid geometry
- Verify texture coordinates exist (vt lines in OBJ)
- Enable `--debug` to see detailed processing information

### Performance Tips

1. **For faster processing**: Use lower output sizes (4096 or 2048)
2. **For better quality**: Use 8192 or higher with more processing time
3. **For batch jobs**: Process during off-hours, use `--max-files` for testing
4. **For debugging**: Always use `--debug` flag to see detailed progress

## 📝 Citation

If you use this tool in research, please cite:

```bibtex
@software{solar_panel_orthomosaic,
  title={Solar Panel Orthomosaic Generator: ICPR-Ready Pattern Preservation Algorithm},
  author={UditKandpal},
  year={2024},
  description={High-quality orthomosaic generation optimized for solar panel installations}
}
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- BVH/Octree spatial acceleration
- GPU rendering support
- Additional output formats (GeoTIFF)
- Advanced camera calibration
- Real-time preview capabilities

## 📄 License

This project is provided as-is for research and commercial use.

## 🔗 Additional Resources

- **OBJ Format Specification**: https://en.wikipedia.org/wiki/Wavefront_.obj_file
- **Pattern Recognition Metrics**: See ICPR conference proceedings
- **Photogrammetry Software**: Agisoft Metashape, WebODM, DroneDeploy

## 💡 Tips for Best Results

1. **Input Quality**: Use high-resolution textures (2048x2048 or larger)
2. **Model Preparation**: Ensure clean topology and proper UV mapping
3. **Output Size**: Match to your analysis needs (6144px is a good default)
4. **Comparison**: Always compare with metrics tool to validate improvements
5. **Iteration**: Try different settings and compare quantitatively

---

**Author**: UditKandpal  
**Last Updated**: December 2024  
**Version**: 1.0