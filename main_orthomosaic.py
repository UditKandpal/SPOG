import os
import sys
import argparse
import glob
import json
from datetime import datetime
import time
from pathlib import Path

# Import your orthomosaic generator
from solar_panel_orthomosaic import generate_enhanced_orthomosaic, SolarPanelICPRGenerator

class OrthomosaicBatchProcessor:
    """Batch processor for generating orthomosaics from multiple OBJ files"""
    
    def __init__(self, base_path, output_base_path, debug=True):
        self.base_path = Path(base_path)
        self.output_base_path = Path(output_base_path)
        self.debug = debug
        self.results = {}
        
        # Create output directory
        self.output_base_path.mkdir(parents=True, exist_ok=True)
        
    def find_obj_files(self):
        """Find all OBJ files in the base directory and subdirectories"""
        obj_files = list(self.base_path.rglob("*.obj"))
        print(f"Found {len(obj_files)} OBJ files in {self.base_path}")
        return obj_files
    
    def find_associated_files(self, obj_file):
        """Find associated MTL and texture files for an OBJ file"""
        obj_path = Path(obj_file)
        base_name = obj_path.stem
        obj_dir = obj_path.parent
        
        # Find MTL file
        mtl_files = [
            obj_path.with_suffix('.mtl'),
            obj_dir / f"{base_name}.mtl",
            obj_dir / "textures" / f"{base_name}.mtl"
        ]
        
        mtl_file = None
        for mtl in mtl_files:
            if mtl.exists():
                mtl_file = mtl
                break
        
        # Find texture files (PNG, JPG, JPEG)
        texture_files = []
        
        # Common texture directories
        texture_dirs = [
            obj_dir,
            obj_dir / "textures",
            obj_dir / "Texture",
            obj_dir / "images",
            obj_dir / "Textures"
        ]
        
        # Common texture extensions
        texture_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tga', '*.bmp']
        
        for texture_dir in texture_dirs:
            if texture_dir.exists():
                for ext in texture_extensions:
                    texture_files.extend(texture_dir.glob(ext))
                    texture_files.extend(texture_dir.glob(ext.upper()))
        
        # Remove duplicates and non-existent files
        texture_files = list(set([f for f in texture_files if f.exists()]))
        
        # Also check for textures referenced in MTL file
        if mtl_file and mtl_file.exists():
            additional_textures = self.extract_textures_from_mtl(mtl_file)
            for tex in additional_textures:
                if tex.exists() and tex not in texture_files:
                    texture_files.append(tex)
        
        # Find image files for EXIF data (if available)
        image_files = []
        image_dirs = [obj_dir / "images", obj_dir / "Photos", obj_dir / "originals"]
        image_extensions = ['*.jpg', '*.jpeg', '*.tiff', '*.tif']
        
        for img_dir in image_dirs:
            if img_dir.exists():
                for ext in image_extensions:
                    image_files.extend(img_dir.glob(ext))
                    image_files.extend(img_dir.glob(ext.upper()))
        
        image_files = list(set([f for f in image_files if f.exists()]))
        
        return {
            'obj_file': obj_path,
            'mtl_file': mtl_file,
            'texture_files': texture_files,
            'image_files': image_files
        }
    
    def extract_textures_from_mtl(self, mtl_file):
        """Extract texture file paths from MTL file"""
        texture_files = []
        try:
            with open(mtl_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('map_Kd'):
                        texture_path = line.split(' ', 1)[1].strip()
                        # Handle both relative and absolute paths
                        possible_paths = [
                            mtl_file.parent / texture_path,
                            mtl_file.parent / "textures" / texture_path,
                            mtl_file.parent / "Texture" / texture_path,
                        ]
                        for path in possible_paths:
                            if path.exists():
                                texture_files.append(path)
                                break
        except Exception as e:
            if self.debug:
                print(f"    Warning: Could not parse MTL file {mtl_file}: {e}")
        
        return texture_files
    
    def create_output_structure(self, obj_file):
        """Create organized output directory structure"""
        obj_path = Path(obj_file)
        relative_path = obj_path.relative_to(self.base_path) if self.base_path in obj_path.parents else obj_path.name
        
        # Create output directory that mirrors input structure
        output_dir = self.output_base_path / relative_path.parent / obj_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir
    
    def validate_file_set(self, file_set):
        """Validate that we have the necessary files for processing"""
        obj_file = file_set['obj_file']
        
        if not obj_file.exists():
            return False, f"OBJ file does not exist: {obj_file}"
        
        if not file_set['texture_files']:
            print(f"  Warning: No texture files found for {obj_file.name}")
            # We'll still try to process, but results may be limited
        
        return True, "Valid"
    
    def process_single_obj(self, obj_file, output_size=6144, force_reprocess=False):
        """Process a single OBJ file and generate orthomosaic"""
        print(f"\n{'='*80}")
        print(f"Processing: {obj_file}")
        print(f"{'='*80}")
        
        # Find associated files
        file_set = self.find_associated_files(obj_file)
        
        # Validate file set
        is_valid, message = self.validate_file_set(file_set)
        if not is_valid:
            print(f"  Skipping: {message}")
            return False
        
        # Create output directory
        output_dir = self.create_output_structure(obj_file)
        output_path = output_dir / f"{Path(obj_file).stem}_orthomosaic.png"
        
        # Check if already processed
        if output_path.exists() and not force_reprocess:
            print(f"  Output already exists: {output_path}")
            print(f"  Use --force to reprocess")
            return True
        
        print(f"  OBJ file: {file_set['obj_file']}")
        print(f"  MTL file: {file_set['mtl_file']}")
        print(f"  Textures: {len(file_set['texture_files'])} files")
        print(f"  Images: {len(file_set['image_files'])} files")
        print(f"  Output: {output_path}")
        
        # Display texture files
        if file_set['texture_files']:
            print("  Texture files:")
            for tex in file_set['texture_files'][:5]:  # Show first 5
                print(f"    - {tex.name}")
            if len(file_set['texture_files']) > 5:
                print(f"    ... and {len(file_set['texture_files']) - 5} more")
        
        # Generate orthomosaic
        start_time = time.time()
        
        success, result = generate_enhanced_orthomosaic(
            obj_file=str(file_set['obj_file']),
            texture_files=[str(tex) for tex in file_set['texture_files']],
            output_path=str(output_path),
            image_files=[str(img) for img in file_set['image_files']],
            image_size=output_size
        )
        
        processing_time = time.time() - start_time
        
        # Store results
        self.results[str(obj_file)] = {
            'success': success,
            'processing_time': processing_time,
            'output_path': str(output_path),
            'file_set': {
                'obj_file': str(file_set['obj_file']),
                'mtl_file': str(file_set['mtl_file']),
                'texture_count': len(file_set['texture_files']),
                'image_count': len(file_set['image_files'])
            }
        }
        
        if success:
            print(f"\n✓ Successfully processed {obj_file.name}")
            print(f"  Time: {processing_time:.2f} seconds")
            print(f"  Output: {output_path}")
            
            # Print pattern metrics summary
            if "Pattern Recognition Metrics" in result:
                lines = result.split('\n')
                for line in lines:
                    if any(indicator in line for indicator in ['Edge Density', 'High Frequency', 'Texture Uniformity', 'Line Density']):
                        print(f"  {line.strip()}")
        else:
            print(f"\n✗ Failed to process {obj_file.name}")
            print(f"  Error: {result}")
        
        return success
    
    def process_batch(self, output_size=6144, force_reprocess=False, max_files=None):
        """Process all OBJ files in batch"""
        print("Starting batch processing...")
        print(f"Input directory: {self.base_path}")
        print(f"Output directory: {self.output_base_path}")
        print(f"Output size: {output_size}px")
        print(f"Force reprocess: {force_reprocess}")
        print(f"Max files: {max_files if max_files else 'Unlimited'}")
        
        # Find all OBJ files
        obj_files = self.find_obj_files()
        
        if not obj_files:
            print("No OBJ files found!")
            return False
        
        if max_files:
            obj_files = obj_files[:max_files]
            print(f"Processing first {max_files} files")
        
        total_files = len(obj_files)
        successful = 0
        failed = 0
        
        # Process each file
        for i, obj_file in enumerate(obj_files, 1):
            print(f"\n[{i}/{total_files}] ", end="")
            
            try:
                if self.process_single_obj(obj_file, output_size, force_reprocess):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  Unexpected error processing {obj_file}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
        
        # Generate summary report
        self.generate_summary_report(successful, failed, total_files)
        
        return failed == 0
    
    def generate_summary_report(self, successful, failed, total_files):
        """Generate a comprehensive summary report"""
        report_path = self.output_base_path / "processing_report.json"
        summary_path = self.output_base_path / "processing_summary.txt"
        
        # JSON report with detailed results
        report = {
            'timestamp': datetime.now().isoformat(),
            'input_directory': str(self.base_path),
            'output_directory': str(self.output_base_path),
            'summary': {
                'total_files': total_files,
                'successful': successful,
                'failed': failed,
                'success_rate': (successful / total_files * 100) if total_files > 0 else 0
            },
            'results': self.results,
            'processing_parameters': {
                'algorithm': 'SolarPanelICPRGenerator',
                'version': '1.0'
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Human-readable summary
        with open(summary_path, 'w') as f:
            f.write("ORTHOMOSAIC BATCH PROCESSING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input Directory: {self.base_path}\n")
            f.write(f"Output Directory: {self.output_base_path}\n\n")
            
            f.write("SUMMARY STATISTICS:\n")
            f.write(f"  Total OBJ files processed: {total_files}\n")
            f.write(f"  Successful: {successful}\n")
            f.write(f"  Failed: {failed}\n")
            f.write(f"  Success rate: {report['summary']['success_rate']:.1f}%\n\n")
            
            if successful > 0:
                f.write("SUCCESSFUL PROCESSING RESULTS:\n")
                for obj_file, result in self.results.items():
                    if result['success']:
                        f.write(f"  {Path(obj_file).name}:\n")
                        f.write(f"    Time: {result['processing_time']:.2f}s\n")
                        f.write(f"    Output: {Path(result['output_path']).relative_to(self.output_base_path)}\n")
                        f.write(f"    Textures: {result['file_set']['texture_count']}\n")
                        f.write(f"    Images: {result['file_set']['image_count']}\n\n")
            
            if failed > 0:
                f.write("FAILED PROCESSES:\n")
                for obj_file, result in self.results.items():
                    if not result['success']:
                        f.write(f"  {Path(obj_file).name}\n")
        
        print(f"\n{'='*80}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Total files: {total_files}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {report['summary']['success_rate']:.1f}%")
        print(f"\nDetailed report: {report_path}")
        print(f"Summary: {summary_path}")

def main():
    """Main command-line interface"""
    parser = argparse.ArgumentParser(
        description='Generate solar panel optimized orthomosaics from OBJ files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Process all OBJ files in current directory
  python main_orthomosaic.py --input . --output ./orthomosaics
  
  # Process specific OBJ file
  python main_orthomosaic.py --input model.obj --output ./output
  
  # Process with custom size and force reprocess
  python main_orthomosaic.py --input ./models --output ./results --size 8192 --force --max-files 10
        '''
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Input OBJ file or directory containing OBJ files')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory for orthomosaics')
    parser.add_argument('--size', '-s', type=int, default=6144,
                       help='Output image size (longest side, default: 6144)')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Force reprocessing even if output exists')
    parser.add_argument('--max-files', '-m', type=int,
                       help='Maximum number of files to process')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Enable debug output')
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {args.input}")
        sys.exit(1)
    
    # Initialize processor
    processor = OrthomosaicBatchProcessor(
        base_path=input_path if input_path.is_dir() else input_path.parent,
        output_base_path=args.output,
        debug=args.debug
    )
    
    # Process files
    if input_path.is_file() and input_path.suffix.lower() == '.obj':
        # Single file processing
        success = processor.process_single_obj(
            input_path, 
            output_size=args.size, 
            force_reprocess=args.force
        )
        if not success:
            sys.exit(1)
    else:
        # Batch processing
        success = processor.process_batch(
            output_size=args.size,
            force_reprocess=args.force,
            max_files=args.max_files
        )
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main()