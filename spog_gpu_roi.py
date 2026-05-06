"""
SPOG GPU — Panel ROI Hard-Boundary Contouring
==========================================

Novel contribution: automatic detection of the solar array footprint in the
rendered orthomosaic, followed by a Gaussian-feathered soft alpha mask that
tapers from full opacity (inside the panel region) to transparent (outside).

Why this is novel:
  Standard photogrammetry outputs include significant background area
  (vegetation, access roads, empty ground) that is irrelevant for defect
  inspection and pollutes downstream metrics.  Existing tools require manual
  annotation of the array boundary.  PanelROIMasker derives this boundary
  automatically by exploiting the known radiometric properties of PV modules:
    • Solar panels are spectrally "dark" (low reflectance in visible) with low
      colour saturation compared to vegetation or bare soil.
    • Vegetation occupies a well-defined HSV hue band (green channel dominant).
    • The rendered mesh boundary (black pixels) is trivially excluded.
  Morphological closing bridges the gaps between panel rows; Gaussian feathering
  (σ = FEATHER_SIGMA) replaces the hard polygon crop used in manual workflows
  with a soft, publication-quality taper.

Output:
  • ortho_masked  : RGB array where background is grey (BACKGROUND_FILL)
  • alpha         : uint8 [0,255] soft alpha mask (255 = full panel)
  • stats         : dict of coverage metrics
"""

import cv2
import numpy as np


# ── Tuneable parameters ────────────────────────────────────────────────────────
FEATHER_SIGMA     = 3      # Gaussian σ (px) — near-hard boundary contour (was 40)
MORPH_CLOSE_PX    = 60     # Morphological close kernel (bridges panel row gaps)
MIN_PANEL_AREA    = 5_000  # Minimum connected-component area (px²) to keep
VEG_HUE_LO       = 30     # Lower HSV hue bound for vegetation mask (°/2 in OpenCV)
VEG_HUE_HI       = 85     # Upper HSV hue bound for vegetation mask
VEG_SAT_MIN      = 40     # Minimum saturation for vegetation (out of 255)
VEG_VAL_MIN      = 40     # Minimum value for vegetation (out of 255)
BACKGROUND_FILL   = 0      # Outside panel region → pure black (was 240 / near-white)


class PanelROIMasker:
    """
    Detect and softly mask the solar panel region in an orthomosaic.

    Usage
    -----
    masker = PanelROIMasker(debug=True)
    ortho_masked, roi_meta = masker.apply(ortho_rgb)

    ortho_masked : np.ndarray uint8 (H, W, 3) — background replaced with grey
    roi_meta     : dict with keys 'alpha', 'stats', 'panel_coverage_pct'
    """

    def __init__(self, debug: bool = True):
        self.debug = debug

    # ── Step 1: rendered-pixel mask ────────────────────────────────────────────

    def _rendered_mask(self, ortho: np.ndarray) -> np.ndarray:
        """True where the orthomosaic has actual rendered content (not black)."""
        gray = cv2.cvtColor(ortho, cv2.COLOR_RGB2GRAY)
        return gray > 8

    # ── Step 2: vegetation mask ────────────────────────────────────────────────

    def _vegetation_mask(self, ortho: np.ndarray) -> np.ndarray:
        """
        Identify vegetation pixels using HSV colour thresholds.

        Solar panels:   low saturation, moderate-dark value, blue/grey hue
        Vegetation:     green hue (H ≈ 60–170° → OpenCV H ≈ 30–85),
                        significant saturation, moderate-to-high value
        """
        hsv = cv2.cvtColor(ortho, cv2.COLOR_RGB2HSV)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        veg = (
            (H >= VEG_HUE_LO) & (H <= VEG_HUE_HI) &
            (S >= VEG_SAT_MIN) &
            (V >= VEG_VAL_MIN)
        )
        return veg.astype(bool)

    # ── Step 3: panel binary mask ──────────────────────────────────────────────

    def _panel_binary_mask(self, ortho: np.ndarray) -> np.ndarray:
        """
        Combine rendered mask and vegetation exclusion to get panel pixels.

        1. Start with all rendered pixels.
        2. Remove vegetation.
        3. Morphological close (MORPH_CLOSE_PX) to bridge inter-row gaps.
        4. Remove small isolated blobs (< MIN_PANEL_AREA px).
        5. Final binary mask.
        """
        rendered = self._rendered_mask(ortho)
        veg      = self._vegetation_mask(ortho)

        # Panel candidates: rendered and not vegetation
        panel_raw = rendered & (~veg)

        # Morphological close: bridge gaps between panel rows
        k    = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MORPH_CLOSE_PX, MORPH_CLOSE_PX)
        )
        closed = cv2.morphologyEx(
            panel_raw.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k
        ) > 127

        # Remove small connected components
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            closed.astype(np.uint8), connectivity=8
        )
        clean = np.zeros_like(closed, dtype=bool)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= MIN_PANEL_AREA:
                clean |= (labels == lbl)

        if self.debug:
            n_comp = int((np.unique(labels[clean]) > 0).sum())
            print(f"  ROI: {n_comp} panel component(s) found  "
                  f"(coverage {clean.sum() / clean.size * 100:.1f}%)")

        return clean

    # ── Step 4: soft alpha mask ────────────────────────────────────────────────

    def _make_alpha(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Convert a binary panel mask into a smooth [0, 255] alpha channel.

        The interior of the solar array is fully opaque (255).
        The boundary tapers to 0 over FEATHER_SIGMA pixels via Gaussian blur,
        replacing the hard polygon crop with a soft publication-quality taper.
        """
        # Gaussian blur on the float binary mask → soft [0, 1] weight
        soft  = cv2.GaussianBlur(
            binary_mask.astype(np.float32), (0, 0), FEATHER_SIGMA
        )
        # Re-normalise so the peak (interior) stays at 1.0
        if soft.max() > 0:
            soft = soft / soft.max()
        alpha = np.clip(soft * 255, 0, 255).astype(np.uint8)
        return alpha

    # ── Public API ─────────────────────────────────────────────────────────────

    def apply(self, ortho: np.ndarray) -> tuple:
        """
        Detect the panel ROI and apply a soft alpha mask.

        Parameters
        ----------
        ortho : np.ndarray (H, W, 3) uint8 — input RGB orthomosaic

        Returns
        -------
        ortho_masked : np.ndarray (H, W, 3) uint8
            Background replaced with BACKGROUND_FILL; panel area unchanged.
        roi_meta : dict
            'alpha'               : np.ndarray (H, W) uint8 soft alpha [0–255]
            'panel_coverage_pct'  : float — percentage of canvas that is panel
            'stats'               : dict with additional statistics
        """
        H, W = ortho.shape[:2]

        binary = self._panel_binary_mask(ortho)
        alpha  = self._make_alpha(binary)

        # Hard binary contour: pixels inside the panel region are kept as-is;
        # pixels outside are set to pure black with no blending/feathering.
        # This produces a sharp, publication-quality contour edge.
        ortho_masked = ortho.copy()
        ortho_masked[~binary] = 0

        coverage = float(binary.sum() / binary.size * 100)
        stats = {
            'panel_pixels':        int(binary.sum()),
            'total_pixels':        int(H * W),
            'panel_coverage_pct':  round(coverage, 2),
            'feather_sigma_px':    FEATHER_SIGMA,
            'morph_close_px':      MORPH_CLOSE_PX,
            'background_fill':     BACKGROUND_FILL,
        }

        roi_meta = {
            'alpha':              alpha,
            'panel_coverage_pct': coverage,
            'stats':              stats,
        }
        return ortho_masked, roi_meta

    # ── Standalone debug helper ────────────────────────────────────────────────

    def save_debug_layers(self, ortho: np.ndarray, out_dir: str) -> None:
        """
        Save intermediate masks for visual inspection / paper figures.

        Files written:
          debug_rendered_mask.png   — all rendered pixels (white)
          debug_vegetation_mask.png — detected vegetation (green overlay)
          debug_panel_binary.png    — binary panel mask (white)
          debug_alpha_mask.png      — soft alpha mask (greyscale)
          debug_composite.png       — final masked orthomosaic
        """
        import os
        from PIL import Image

        os.makedirs(out_dir, exist_ok=True)

        rendered = self._rendered_mask(ortho)
        veg      = self._vegetation_mask(ortho)
        binary   = self._panel_binary_mask(ortho)
        alpha    = self._make_alpha(binary)

        # Rendered mask
        Image.fromarray((rendered * 255).astype(np.uint8)).save(
            os.path.join(out_dir, 'debug_rendered_mask.png'))

        # Vegetation overlay (green tint on original)
        veg_vis = ortho.copy()
        veg_vis[veg] = [0, 200, 0]
        Image.fromarray(veg_vis).save(
            os.path.join(out_dir, 'debug_vegetation_mask.png'))

        # Binary panel mask
        Image.fromarray((binary * 255).astype(np.uint8)).save(
            os.path.join(out_dir, 'debug_panel_binary.png'))

        # Soft alpha mask
        Image.fromarray(alpha).save(
            os.path.join(out_dir, 'debug_alpha_mask.png'))

        # Final composite
        ortho_masked, _ = self.apply(ortho)
        Image.fromarray(ortho_masked).save(
            os.path.join(out_dir, 'debug_composite.png'))

        print(f"  Debug layers saved to {out_dir}/")
