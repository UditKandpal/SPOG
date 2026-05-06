"""
SPOG GPU Runner
===============
Generates a high-fidelity GPU-rendered orthomosaic from a textured OBJ mesh,
optimised for UAV solar farm survey outputs (WebODM, OpenDroneMap, Agisoft).

Usage:
    python3 spog_gpu_runner.py --model <folder>   # path to folder with OBJ + textures
    python3 spog_gpu_runner.py --model model_1    # use bundled model_1 example
    python3 spog_gpu_runner.py --size 8192        # override output resolution
    python3 spog_gpu_runner.py --no-roi           # skip panel ROI masking

Outputs (in results/<model_name>/):
    spog_gpu_orthomosaic.tif         — 16-bit uncompressed TIFF (primary output)
    spog_gpu_orthomosaic_masked.tif  — RGBA 16-bit TIFF with hard ROI alpha
    spog_gpu_orthomosaic_preview.png — 8-bit PNG preview (opens in any viewer)
    spog_gpu_orthomosaic_metadata.json
    debug_roi/                       — ROI intermediate masks (optional, --debug-roi)

Requirements:
    pip install moderngl numpy pillow opencv-python scipy tifffile
    NVIDIA GPU with EGL support (tested on RTX A4000)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from spog_gpu     import SPOGGPUGenerator
from spog_gpu_roi import PanelROIMasker

BASE_DIR    = Path(__file__).parent          # repo root (or working directory)
RESULTS_DIR = BASE_DIR / 'results'


def find_textures(model_dir: Path) -> list[str]:
    exts = {'.png', '.jpg', '.jpeg', '.tga', '.bmp', '.tif', '.tiff'}
    return sorted([str(f) for f in model_dir.iterdir()
                   if f.suffix.lower() in exts])


def main():
    parser = argparse.ArgumentParser(description="SPOG GPU — Solar Farm Orthomosaic Generator")
    parser.add_argument('--model',     default='model_1',
                        help='Model folder containing OBJ + textures (default: model_1)')
    parser.add_argument('--obj',       default=None,
                        help='Explicit path to .obj file (auto-detected if omitted)')
    parser.add_argument('--size',      type=int, default=16384,
                        help='Maximum output side length in pixels (default: 16384)')
    parser.add_argument('--no-roi',    action='store_true',
                        help='Skip panel ROI hard-boundary masking')
    parser.add_argument('--debug-roi', action='store_true',
                        help='Save intermediate ROI mask layers to debug_roi/')
    args = parser.parse_args()

    model_dir  = Path(args.model) if Path(args.model).is_absolute() else BASE_DIR / args.model
    output_dir = RESULTS_DIR / model_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve OBJ: explicit --obj flag > well-known ODM name > any .obj in folder
    if args.obj:
        obj_file = Path(args.obj)
    else:
        candidate = model_dir / 'drm_textured_model_geo.obj'
        if candidate.exists():
            obj_file = candidate
        else:
            objs = list(model_dir.glob('*.obj'))
            if not objs:
                print(f"ERROR: No .obj file found in {model_dir}")
                sys.exit(1)
            if len(objs) > 1:
                print(f"WARNING: Multiple .obj files found; using {objs[0].name}. "
                      f"Use --obj to specify explicitly.")
            obj_file = objs[0]

    output_tif = output_dir / 'spog_gpu_orthomosaic.tif'

    print(f"Model   : {model_dir}")
    print(f"OBJ     : {obj_file.name}")
    print(f"Output  : {output_dir}  (16-bit uncompressed TIFF)")
    print(f"Canvas  : up to {args.size} px (longest side)")

    if not obj_file.exists():
        print(f"ERROR: OBJ not found at {obj_file}")
        sys.exit(1)

    textures = find_textures(model_dir)
    if not textures:
        print(f"ERROR: No texture files found in {model_dir}")
        sys.exit(1)
    print(f"Textures: {len(textures)} file(s)")

    gen     = SPOGGPUGenerator(debug=True)
    t0      = time.time()
    ok, msg = gen.generate(
        obj_file       = str(obj_file),
        texture_files  = textures,
        output_path    = str(output_tif),
        image_size     = args.size,
        apply_roi_mask = not args.no_roi,
    )
    elapsed = time.time() - t0

    if not ok:
        print(f"\nFAILED: {msg}")
        sys.exit(1)

    print(f"\nOutput saved to : {output_tif}")
    print(f"Total wall time : {elapsed:.1f}s")

    if args.debug_roi and not args.no_roi:
        import numpy as np
        import tifffile
        raw    = tifffile.imread(str(output_tif))
        ortho  = (raw // 257).astype('uint8')
        masker = PanelROIMasker(debug=True)
        masker.save_debug_layers(ortho, str(output_dir / 'debug_roi'))

    print("\nDone.")
    print(f"  16-bit TIFF     : {output_tif}")
    print(f"  RGBA TIFF       : {str(output_tif).replace('.tif', '_masked.tif')}")
    print(f"  PNG preview     : {str(output_tif).replace('.tif', '_preview.png')}")
    print(f"  Metadata        : {str(output_tif).replace('.tif', '_metadata.json')}")


if __name__ == '__main__':
    main()
