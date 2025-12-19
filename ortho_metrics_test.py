import os
import cv2
import numpy as np
from scipy import ndimage
from scipy.fft import fft2, fftshift
from skimage import metrics, feature, measure, filters
import math
from sklearn.cluster import KMeans
import argparse
from scipy.signal import find_peaks

def ensure_same_dimensions(img1, img2, method='resize'):
    """
    Make two images have the same dimensions.
    Options:
        'resize': Resize the smaller image to match the larger one (default, fast)
        'crop': Crop both images to the overlapping center region (more fair for metrics)
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if h1 == h2 and w1 == w2:
        return img1.copy(), img2.copy()

    if method == 'crop':
        # Crop both to smallest common size (centered)
        target_h = min(h1, h2)
        target_w = min(w1, w2)

        start_y1 = (h1 - target_h) // 2
        start_x1 = (w1 - target_w) // 2
        start_y2 = (h2 - target_h) // 2
        start_x2 = (w2 - target_w) // 2

        img1_cropped = img1[start_y1:start_y1 + target_h, start_x1:start_x1 + target_w]
        img2_cropped = img2[start_y2:start_y2 + target_h, start_x2:start_x2 + target_w]

        return img1_cropped, img2_cropped

    else:  # 'resize' (default)
        target_h, target_w = max(h1, h2), max(w1, w2)

        if h1 != target_h or w1 != target_w:
            img1 = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        if h2 != target_h or w2 != target_w:
            img2 = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        return img1, img2


class OrthomosaicMetrics:
    """Calculate pattern preservation and quality metrics for orthomosaic comparison"""
    
    def __init__(self):
        self.all_results = []
        self.image_names = []
    
    def calculate_metrics_single(self, gray_image):
        """Calculate all metrics for a single image"""
        
        results = {}
        
        # Edge Density
        median = np.median(gray_image)
        lower = max(0, 0.7 * median)
        upper = min(255, 1.3 * median)
        edges = feature.canny(gray_image, low_threshold=lower, high_threshold=upper)
        results['edge_density'] = float(np.sum(edges) / edges.size)
        
        # High Frequency Energy
        h, w = gray_image.shape
        window = np.outer(np.hanning(h), np.hanning(w))
        f = fftshift(np.abs(fft2(gray_image.astype(float) * window)))
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        mask = (x*x + y*y) > (min(cx, cy) * 0.3)**2
        results['high_freq_energy'] = float(np.sum(f * mask))
        
        # Line Density
        edges_for_lines = feature.canny(gray_image, sigma=2)
        edges_uint8 = (edges_for_lines * 255).astype(np.uint8)
        lines = cv2.HoughLinesP(edges_uint8, 1, np.pi/180, threshold=30,
                                minLineLength=30, maxLineGap=15)
        results['line_count'] = len(lines) if lines is not None else 0
        
        # Texture Uniformity
        local_std = ndimage.generic_filter(gray_image.astype(float), np.std, size=7, mode='reflect')
        results['texture_uniformity'] = float(1.0 / (1.0 + np.mean(local_std)))
        
        # Edge Regularity
        gx = filters.sobel_v(gray_image.astype(float))
        gy = filters.sobel_h(gray_image.astype(float))
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx)
        threshold = np.percentile(magnitude, 85)
        strong = magnitude > threshold
        if np.any(strong):
            angles = orientation[strong]
            mean_vec = np.mean(np.exp(1j * angles))
            results['edge_regularity'] = float(1 - np.abs(mean_vec))
        else:
            results['edge_regularity'] = 1.0
        
        # Global Contrast
        results['contrast'] = float(np.std(gray_image))
        
        # Panel-like Structures
        contours = measure.find_contours(edges_for_lines, 0.8)
        panel_count = 0
        for contour in contours:
            if len(contour) < 20:
                continue
            hull = cv2.convexHull(contour.astype(np.float32))
            if len(hull) < 4:
                continue
            rect = cv2.minAreaRect(hull)
            (w_rect, h_rect) = rect[1]
            if w_rect < 20 or h_rect < 20:
                continue
            aspect = max(w_rect, h_rect) / min(w_rect, h_rect)
            if 1.3 <= aspect <= 3.5:
                panel_count += 1
        results['panel_structures'] = panel_count
        
        # Pattern Regularity
        norm = (gray_image - gray_image.mean()) / (gray_image.std() + 1e-8)
        corr = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(np.fft.fft2(norm))**2)))
        corr = corr / (corr.max() + 1e-8)
        center = corr[corr.shape[0]//2, corr.shape[1]//2]
        corr[corr.shape[0]//2, corr.shape[1]//2] = 0
        peaks, _ = find_peaks(corr.flatten(), height=0.3 * center)
        if len(peaks) >= 3:
            distances = []
            cy_corr, cx_corr = corr.shape[0]//2, corr.shape[1]//2
            for p in peaks:
                y_p, x_p = np.unravel_index(p, corr.shape)
                distances.append(np.sqrt((y_p-cy_corr)**2 + (x_p-cx_corr)**2))
            if len(distances) >= 2:
                std_dist = np.std(distances)
                results['pattern_regularity'] = float(1.0 / (1.0 + std_dist))
            else:
                results['pattern_regularity'] = 0.0
        else:
            results['pattern_regularity'] = 0.0
        
        return results
    
    def add_image(self, image, name):
        """Add an image to the comparison"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        results = self.calculate_metrics_single(gray)
        self.all_results.append(results)
        self.image_names.append(name)
    
    def generate_multi_comparison_report(self, save_path=None):
        """Generate comparison table for all images"""
        if not self.all_results:
            return "No results to report."
        
        num_images = len(self.all_results)
        
        # Calculate column width based on number of images
        name_col_width = 45
        value_col_width = 14
        
        report = "ORTHOMOSAIC QUANTITATIVE COMPARISON - ALL METHODS\n"
        report += "=" * (name_col_width + num_images * value_col_width + 4) + "\n"
        
        # Header row
        header = f"{'Metric':<{name_col_width}}"
        for name in self.image_names:
            header += f"{name:>{value_col_width}}"
        report += header + "\n"
        report += "-" * (name_col_width + num_images * value_col_width + 4) + "\n"
        
        # Metrics rows
        metrics_info = [
            ('Edge Density (edges/px)', 'edge_density', 6, 1.0),
            ('High-Frequency Energy (×10⁹)', 'high_freq_energy', 2, 1e-9),
            ('Detected Grid Lines', 'line_count', 0, 1.0),
            ('Texture Uniformity (lower = less smoothed)', 'texture_uniformity', 4, 1.0),
            ('Edge Orientation Regularity (lower better)', 'edge_regularity', 4, 1.0),
            ('Global Contrast (std dev)', 'contrast', 2, 1.0),
            ('Panel-like Structures Detected', 'panel_structures', 0, 1.0),
            ('Pattern Regularity Score', 'pattern_regularity', 4, 1.0),
        ]
        
        for metric_name, metric_key, precision, scale in metrics_info:
            row = f"{metric_name:<{name_col_width}}"
            values = []
            for result in self.all_results:
                val = result[metric_key] * scale
                values.append(val)
                row += f"{val:>{value_col_width}.{precision}f}"
            report += row + "\n"
            
            # Add improvement analysis row
            if num_images > 1:
                # Find best and worst (considering if lower is better for some metrics)
                lower_is_better = metric_key in ['texture_uniformity', 'edge_regularity']
                
                if lower_is_better:
                    best_idx = values.index(min(values))
                    worst_idx = values.index(max(values))
                else:
                    best_idx = values.index(max(values))
                    worst_idx = values.index(min(values))
                
                improvement_row = f"{'  └─ Best: ' + self.image_names[best_idx]:<{name_col_width}}"
                
                for i, val in enumerate(values):
                    if i == best_idx:
                        improvement_row += f"{'BEST':>{value_col_width}}"
                    elif values[best_idx] != 0:
                        if lower_is_better:
                            diff_pct = (val - values[best_idx]) / values[best_idx] * 100
                            improvement_row += f"{diff_pct:>+{value_col_width-1}.1f}%"
                        else:
                            diff_pct = (val - values[best_idx]) / values[best_idx] * 100
                            improvement_row += f"{diff_pct:>+{value_col_width-1}.1f}%"
                    else:
                        improvement_row += f"{'N/A':>{value_col_width}}"
                
                report += improvement_row + "\n"
        
        report += "-" * (name_col_width + num_images * value_col_width + 4) + "\n"
        report += "Note: For metrics marked 'lower better', negative % means better than BEST\n"
        report += "=" * (name_col_width + num_images * value_col_width + 4) + "\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Comparison report saved → {save_path}")
        
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare multiple orthomosaic images with advanced metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all 4 methods
  python script.py --images odm.tif dd.tif metashape.tif custom.tif \\
                   --names WebODM DroneDeploy Metashape CustomMethod

  # Compare 2 methods
  python script.py --images img1.tif img2.tif --names Method1 Method2

  # With custom output
  python script.py --images a.tif b.tif c.tif --names A B C --output comparison.txt
        """
    )
    
    parser.add_argument('--images', nargs='+', required=True,
                        help='Paths to orthomosaic images (2 or more)')
    parser.add_argument('--names', nargs='+', required=True,
                        help='Display names for each image (must match number of images)')
    parser.add_argument('--output', type=str, default='multi_comparison_report.txt',
                        help='Output report path')
    parser.add_argument('--method', type=str, choices=['resize', 'crop'], default='resize',
                        help='Method to align images: resize (default) or crop')
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.images) != len(args.names):
        raise ValueError(f"Number of images ({len(args.images)}) must match number of names ({len(args.names)})")
    
    if len(args.images) < 2:
        raise ValueError("At least 2 images required for comparison")
    
    print(f"Loading {len(args.images)} images for comparison...")
    
    # Load all images
    images = []
    for i, img_path in enumerate(args.images):
        print(f"  [{i+1}/{len(args.images)}] Loading {args.names[i]}: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        print(f"       Dimensions: {img.shape[1]}×{img.shape[0]}")
    
    # Align all images to the same dimensions
    print(f"\nAligning all images using '{args.method}' method...")
    
    # Find target dimensions
    if args.method == 'crop':
        target_h = min(img.shape[0] for img in images)
        target_w = min(img.shape[1] for img in images)
    else:  # resize
        target_h = max(img.shape[0] for img in images)
        target_w = max(img.shape[1] for img in images)
    
    print(f"Target dimensions: {target_w}×{target_h}")
    
    # Align all images
    aligned_images = []
    for i, img in enumerate(images):
        if args.method == 'crop':
            h, w = img.shape[:2]
            start_y = (h - target_h) // 2
            start_x = (w - target_w) // 2
            aligned = img[start_y:start_y + target_h, start_x:start_x + target_w]
        else:  # resize
            if img.shape[0] != target_h or img.shape[1] != target_w:
                aligned = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            else:
                aligned = img.copy()
        aligned_images.append(aligned)
        print(f"  {args.names[i]}: {aligned.shape[1]}×{aligned.shape[0]}")
    
    # Calculate metrics for all images
    print("\nComputing metrics for all images...")
    calculator = OrthomosaicMetrics()
    
    for i, (img, name) in enumerate(zip(aligned_images, args.names)):
        print(f"  [{i+1}/{len(aligned_images)}] Processing {name}...")
        calculator.add_image(img, name)
    
    # Generate report
    print("\nGenerating comparison report...")
    report = calculator.generate_multi_comparison_report(args.output)
    
    print("\n" + report)