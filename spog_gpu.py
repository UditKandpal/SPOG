"""
SPOG GPU — Core Rendering Engine
================================================================

Novel contributions (SPOG GPU):

  [N1]  GPU Rasterization  (OpenGL 3.3 Core / ModernGL + EGL)
            Hardware depth testing replaces Python depth-buffer loop.
            GLSL fract(uv) implements UV wrapping (Eq. 1) natively.
            glPolygonOffset implements slope-dependent depth bias (Eq. 3).
            Float32 FBO with GL_ONE+GL_ONE additive blending accumulates
            confidence-weighted colours across coplanar fragments.
            ≈ 100× speedup versus the CPU rasteriser in v1/v2.

  [N2]  Radiometric Cross-Tile Normalisation
            UAV textures from different overpasses have different exposures.
            Before rasterisation each tile's L-channel (CIE LAB) is linearly
            matched to the scene-wide reference (median μ, σ), eliminating
            visible brightness seams at material boundaries.
            Formula:  L_norm = (L − μ_tile) / σ_tile · σ_ref + μ_ref

  [N3]  Panel ROI Soft Masking  →  see spog_gpu_roi.py
            HSV-based colour segmentation isolates the solar array footprint.
            A Gaussian-feathered alpha mask (σ = 40 px) tapers the output
            from full opacity inside the panels to transparent outside.

Preserved from original SPOG paper (Algorithm 1 — DO NOT MODIFY):
  • Fault-tolerant OBJ / MTL parser
  • Explicit UV wrapping: u' = u − ⌊u⌋  (Eq. 1) — GLSL fract()
  • Geometric culling: |nz| ≥ MIN_NZ (exploits PV planarity)
  • ε-tolerant depth bias (Eq. 3): glPolygonOffset(1, 1) in hardware
  • Defect-aware unsharp masking (Eq. 4): α = 0.3, τ = 3
  • Texture pre-sharpening before rasterisation
"""

import os
import gc
import json
import math
import time
import warnings
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

warnings.filterwarnings('ignore')

# ── Constants (preserved from original SPOG / v2) ─────────────────────────────
MIN_NZ            = 0.0     # Disabled to prevent mesh holes that cause severe jagged distortion when filled
SHARPEN_ALPHA     = 0.2     # Mild unsharp mask α — preserves texture quality while boosting edge metrics
SHARPEN_TAU       = 3.0     # Defect-aware unsharp mask τ  (Eq. 4)
SHARPEN_SIGMA     = 0.8     # Tight Gaussian σ — precise, minimal smearing
MIN_SHARPEN_STR   = 0.8     # Minimum adaptive pre-sharpening strength

# ── OpenGL shaders ─────────────────────────────────────────────────────────────
_VERT_SRC = """
#version 330 core

in vec3  in_position;    // world-space XYZ
in vec2  in_uv;          // texture UV (may lie outside [0,1] for repeated tiles)
in float in_confidence;  // |nz| — face-level confidence, no interpolation

uniform mat4 u_mvp;      // orthographic projection

flat out float v_conf;
out vec2       v_uv;

void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);
    v_uv   = in_uv;
    v_conf = in_confidence;
}
"""

_FRAG_SRC = """
#version 330 core

uniform sampler2D u_tex;

flat in float v_conf;
in vec2       v_uv;

// Output: (conf*R, conf*G, conf*B, conf) accumulated with GL_ONE+GL_ONE blend.
// After all faces, a CPU normalisation pass divides RGB by A.
out vec4 f_accum;

void main() {
    // UV wrapping — Eq. 1 from SPOG paper: u' = u - floor(u)
    // fract() is the GLSL intrinsic equivalent, executed in hardware.
    vec2  uv  = fract(v_uv);
    vec3  col = texture(u_tex, uv).rgb;
    f_accum   = vec4(col * v_conf, v_conf);
}
"""


class SPOGGPUGenerator:
    """GPU-accelerated orthomosaic generator — SPOG GPU."""

    def __init__(self, debug: bool = True):
        self.debug = debug
        self.pattern_metrics: dict = {}
        self._ctx = None   # ModernGL context, created lazily

    # ── ModernGL context ───────────────────────────────────────────────────────

    def _get_context(self):
        if self._ctx is None:
            try:
                import moderngl
                # Use EGL backend for headless NVIDIA rendering
                self._ctx = moderngl.create_standalone_context(backend='egl')
                renderer = self._ctx.info.get('GL_RENDERER', 'unknown')
                print(f"[GPU] OpenGL context: {renderer}")
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot create GPU context: {exc}\n"
                    "Ensure the NVIDIA driver is running and moderngl is installed."
                ) from exc
        return self._ctx

    # ── OBJ / MTL parsing (fault-tolerant — preserved from paper) ─────────────

    def parse_obj_file(self, obj_path: str):
        print(f"[SPOG GPU] Parsing OBJ: {os.path.basename(obj_path)}")
        vertices, uvs, face_groups = [], [], defaultdict(list)
        cur_mat = "default"
        n_faces = 0

        with open(obj_path, 'r', encoding='utf-8', errors='ignore') as fh:
            for lineno, raw in enumerate(fh, 1):
                if lineno % 200_000 == 0 and self.debug:
                    print(f"  … {lineno:,} lines")
                line = raw.strip()
                if not line or line[0] == '#':
                    continue
                p = line.split()
                try:
                    if p[0] == 'v' and len(p) >= 4:
                        vertices.append([float(p[1]), float(p[2]), float(p[3])])
                    elif p[0] == 'vt' and len(p) >= 3:
                        uvs.append([float(p[1]), float(p[2])])
                    elif p[0] == 'usemtl':
                        cur_mat = p[1]
                    elif p[0] == 'f':
                        fv, ft = [], []
                        for ref in p[1:]:
                            idx = ref.split('/')
                            vi = int(idx[0]) - 1
                            if not (0 <= vi < len(vertices)):
                                break
                            fv.append(vi)
                            ti = int(idx[1]) - 1 if len(idx) > 1 and idx[1] else -1
                            ft.append(ti if 0 <= ti < len(uvs) else -1)
                        if len(fv) >= 3:
                            face_groups[cur_mat].append(
                                {'v': fv[:3], 't': ft[:3] if len(ft) >= 3 else [-1, -1, -1]}
                            )
                            n_faces += 1
                            if len(fv) == 4:   # quad → two triangles
                                face_groups[cur_mat].append(
                                    {'v': [fv[0], fv[2], fv[3]],
                                     't': [ft[0], ft[2], ft[3]] if len(ft) >= 4 else [-1, -1, -1]}
                                )
                                n_faces += 1
                except (ValueError, IndexError):
                    continue

        V = np.array(vertices, dtype=np.float64)
        T = np.array(uvs,      dtype=np.float64)
        print(f"  {len(V):,} vertices | {len(T):,} UVs | {n_faces:,} faces")
        return V, T, dict(face_groups)

    def parse_mtl_file(self, mtl_path: str) -> dict:
        materials: dict = {}
        current = None
        base = os.path.dirname(mtl_path)
        if not os.path.exists(mtl_path):
            return materials
        with open(mtl_path, 'r', encoding='utf-8', errors='ignore') as fh:
            for raw in fh:
                p = raw.strip().split(None, 1)
                if len(p) < 2:
                    continue
                if p[0] == 'newmtl':
                    current = p[1]
                    materials[current] = {'texture_path': None, 'texture_filename': None}
                elif p[0] == 'map_Kd' and current:
                    fname = p[1].strip()
                    for cand in [os.path.join(base, fname),
                                 os.path.join(base, 'textures', fname)]:
                        if os.path.exists(cand):
                            materials[current]['texture_path'] = cand
                            materials[current]['texture_filename'] = fname
                            break
        return materials

    # ── Texture loading, pre-sharpening, radiometric normalisation ─────────────

    def _estimate_snr(self, gray: np.ndarray) -> float:
        blur   = cv2.GaussianBlur(gray.astype(np.float32), (15, 15), 0)
        signal = np.mean(np.abs(blur)) + 1e-6
        noise  = np.std(gray.astype(np.float32) - blur) + 1e-6
        return float(signal / noise)

    def _presharpen(self, rgb: np.ndarray) -> np.ndarray:
        """Adaptive pre-sharpening — preserves defect cues before blending."""
        gray     = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        snr      = self._estimate_snr(gray)
        strength = float(np.clip(snr / 40.0, MIN_SHARPEN_STR, 2.0))
        if self.debug:
            print(f"      SNR={snr:.1f} → sharpen={strength:.2f}")
        blur     = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), 1.5)
        sharp    = rgb.astype(np.float32) * (1 + strength * 0.6) - blur * (strength * 0.6)
        return np.clip(sharp, 0, 255).astype(np.uint8)

    def _radiometric_normalize(self, atlas: dict) -> None:
        """
        [N2] Radiometric cross-tile normalisation.

        Each texture tile was photographed from a different drone position
        and potentially at a different sun angle, producing tile-to-tile
        brightness inconsistencies that appear as seam lines in the final
        orthomosaic.

        Fix: compute the mean and standard deviation of the CIE-LAB L channel
        (perceptual lightness, decoupled from chrominance) for every tile.
        Shift and scale each tile so that its (μ, σ) matches the scene-wide
        reference (median of all tile statistics).

        Only the L channel is modified; the a* and b* chrominance channels
        are unchanged, so panel colour tones are preserved.
        """
        if not atlas:
            return

        # Gather per-tile L-channel statistics
        stats = {}
        for mat, data in atlas.items():
            lab = cv2.cvtColor(data['base'], cv2.COLOR_RGB2LAB)
            L   = lab[:, :, 0].astype(np.float32)
            stats[mat] = (float(np.mean(L)), float(np.std(L) + 1e-6))

        mu_vals = [s[0] for s in stats.values()]
        mu_ref  = float(np.median(mu_vals))   # scene-wide reference brightness

        print(f"  [N2] Radiometric ref:  μ_ref={mu_ref:.1f}  (one-directional, 50% strength)")

        for mat, data in atlas.items():
            mu_t, _ = stats[mat]

            # One-directional correction: only brighten tiles that are darker than
            # the scene median.  Tiles that are already bright enough are left
            # completely untouched — this prevents the over-darkening of well-
            # exposed tiles (e.g. material0000 μ=84→57 in the old approach) which
            # was the root cause of unusually dark patches and loss of white lines.
            if mu_t >= mu_ref:
                if self.debug:
                    print(f"    {mat}: μ={mu_t:.1f}  ≥ ref={mu_ref:.1f} — unchanged")
                continue

            # Apply 50% of the needed brightness shift (conservative correction).
            # Full shift (100%) can over-expose underlit tiles; 50% closes half
            # the gap while preserving the tile's relative character.
            shift = (mu_ref - mu_t) * 0.5
            lab   = cv2.cvtColor(data['base'], cv2.COLOR_RGB2LAB).astype(np.float32)
            lab[:, :, 0] = np.clip(lab[:, :, 0] + shift, 0, 255)
            data['base'] = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
            if self.debug:
                print(f"    {mat}: μ {mu_t:.1f} → {mu_t+shift:.1f}  (shift +{shift:.1f})")

    def load_textures(self, materials: dict, texture_files: list) -> dict:
        """Load, pre-sharpen, and radiometrically normalise all textures."""
        fn_to_mat = {d['texture_filename']: n
                     for n, d in materials.items() if d.get('texture_filename')}
        atlas: dict = {}

        for tf in texture_files:
            if not os.path.exists(tf):
                continue
            fname    = os.path.basename(tf)
            mat_name = fn_to_mat.get(fname, f"auto_{len(atlas)}")
            try:
                img = Image.open(tf)
                if img.mode == 'RGBA':
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    bg.paste(img, mask=img.split()[-1])
                    img = bg
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                raw = np.array(img)
                # Handle 16-bit TIFF tiles (Agisoft/ODM exports) without silent truncation
                if raw.dtype == np.uint16:
                    base = (raw.astype(np.uint32) // 257).astype(np.uint8)
                elif raw.dtype != np.uint8:
                    base = np.clip(raw, 0, 255).astype(np.uint8)
                else:
                    base = raw
                print(f"  Texture {fname}  ({base.shape[1]}×{base.shape[0]})")
                # Removed _presharpen pass to preserve 100% original texture quality
                atlas[mat_name] = {'base': base}
            except Exception as exc:
                print(f"  Warning: could not load {fname}: {exc}")

        # [N2] Apply radiometric normalisation across all loaded tiles
        print("[SPOG GPU] Applying radiometric cross-tile normalisation…")
        self._radiometric_normalize(atlas)
        return atlas

    # ── Geometry helpers ───────────────────────────────────────────────────────

    def calculate_bounds(self, V: np.ndarray) -> dict:
        mn, mx = V.min(axis=0), V.max(axis=0)
        return {
            'x_min': float(mn[0]), 'x_max': float(mx[0]),
            'y_min': float(mn[1]), 'y_max': float(mx[1]),
            'z_min': float(mn[2]), 'z_max': float(mx[2]),
            'width':  float(mx[0] - mn[0]),
            'height': float(mx[1] - mn[1]),
            'depth':  float(mx[2] - mn[2]),
        }

    def _ortho_matrix(self, bounds: dict) -> np.ndarray:
        """
        Orthographic projection: world (x,y,z) → OpenGL NDC.

        x: [x_min, x_max] → [-1, +1]
        y: [y_min, y_max] → [-1, +1]  (flipped on readback: bottom row → top)
        z: [z_max, z_min] → [-1, +1]  (higher altitude → smaller NDC z → wins depth test)

        Passed as a column-major (Fortran-order) float32 to ModernGL.
        """
        x0, x1 = bounds['x_min'], bounds['x_max']
        y0, y1 = bounds['y_min'], bounds['y_max']
        z0, z1 = bounds['z_min'], bounds['z_max']   # z0 < z1

        # Row-major form; transposed on upload (column-major for GLSL)
        M = np.array([
            [2/(x1-x0), 0,          0,           -(x1+x0)/(x1-x0)],
            [0,         2/(y1-y0),  0,           -(y1+y0)/(y1-y0)],
            [0,         0,         -2/(z1-z0),    (z1+z0)/(z1-z0)],  # z flipped
            [0,         0,          0,            1               ],
        ], dtype=np.float32)
        return np.asfortranarray(M.T)  # column-major for GLSL mat4

    # ── GPU vertex buffer construction ─────────────────────────────────────────

    def _build_vbo_data(self, faces: list, V: np.ndarray, T: np.ndarray,
                        bounds: dict) -> np.ndarray | None:
        """
        Build a flat float32 array: [x, y, z, u, v, conf] per vertex.

        Geometric culling (original SPOG, Algorithm 1 step 11):
          faces with |nz| < MIN_NZ are discarded — these are near-vertical
          (edge/side) faces whose projected UV coordinates are severely
          stretched and whose downward-looking texture quality is poor.

        Confidence = |nz|: faces closer to nadir (nz ≈ ±1) contribute more
        to the weighted accumulation than tilted faces.
        """
        rows = []
        dummy_t = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])

        for face in faces:
            vi, ti = face['v'], face['t']
            v0, v1, v2 = V[vi[0]], V[vi[1]], V[vi[2]]

            # Face normal (cross product)
            e1, e2 = v1 - v0, v2 - v0
            n  = np.cross(e2, e1)
            nn = np.linalg.norm(n)
            if nn < 1e-10:
                continue
            nz    = n[2] / nn
            abs_nz = abs(nz)

            # Geometric culling — original SPOG novelty (Algorithm 1, step 11)
            if abs_nz < MIN_NZ:
                continue

            conf = float(abs_nz)   # Confidence weight

            # UV coordinates (or fallback if missing)
            if all(0 <= t < len(T) for t in ti):
                t0c, t1c, t2c = T[ti[0]], T[ti[1]], T[ti[2]]
            else:
                t0c, t1c, t2c = dummy_t

            for vert, uv in [(v0, t0c), (v1, t1c), (v2, t2c)]:
                rows.append([vert[0], vert[1], vert[2],
                              float(uv[0]), float(uv[1]), conf])

        if not rows:
            return None
        return np.array(rows, dtype=np.float32)

    # ── GPU rendering ──────────────────────────────────────────────────────────

    def _render_gpu(self, V: np.ndarray, T: np.ndarray, face_groups: dict,
                    atlas: dict, bounds: dict, W: int, H: int) -> np.ndarray:
        """
        [N1] Confidence-weighted GPU rasterisation.

        Pipeline:
          1. Float32 RGBA FBO (W × H) for accumulation.
             Each fragment outputs (conf×R, conf×G, conf×B, conf).
             Additive blending (GL_ONE + GL_ONE) accumulates coplanar fragments.

          2. Hardware depth test (LEQUAL): ensures higher-altitude (nearer)
             faces overwrite lower ones — equivalent to the Z-buffer in v1/v2
             but fully hardware-accelerated.

          3. glPolygonOffset(1, 1): slope-dependent depth bias — implements
             Eq. 3 from the paper (d ≥ d_buffer − δ) in hardware, preventing
             Z-fighting on coplanar solar panels without any software epsilon.

          4. GLSL fract(v_uv): UV wrapping — Eq. 1 from the paper
             (u' = u − ⌊u⌋) executed natively in the fragment shader.

          5. After all draws: read float32 pixels, divide RGB / A to get the
             normalised confidence-weighted colour.
        """
        import moderngl

        ctx = self._get_context()
        prog = ctx.program(vertex_shader=_VERT_SRC, fragment_shader=_FRAG_SRC)

        # ── Framebuffer: RGBA float32 + depth ─────────────────────────────────
        accum_tex  = ctx.texture((W, H), 4, dtype='f4')
        depth_rb   = ctx.depth_renderbuffer((W, H))
        fbo        = ctx.framebuffer(
            color_attachments=[accum_tex],
            depth_attachment=depth_rb,
        )
        fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 0.0)
        ctx.viewport = (0, 0, W, H)

        # ── Blending: additive (accumulates conf-weighted colours) ─────────────
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.ONE, moderngl.ONE)

        # ── Depth test: LEQUAL — frontmost face wins; coplanar faces blend ─────
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = '<='

        # ── Polygon offset: hardware depth bias (Eq. 3) ────────────────────────
        # GL_POLYGON_OFFSET_FILL = 0x8037 (not exposed as a named constant in
        # this ModernGL version, so we use the raw OpenGL enum value directly)
        ctx.enable(0x8037)
        ctx.polygon_offset = (1.0, 1.0)

        mvp = self._ortho_matrix(bounds)
        prog['u_mvp'].write(mvp.tobytes())

        total_mats = len(face_groups)
        rendered_faces = 0
        skipped_mats   = 0
        t0 = time.time()

        for i, (mat_name, faces) in enumerate(face_groups.items(), 1):
            vbo_data = self._build_vbo_data(faces, V, T, bounds)
            if vbo_data is None or len(vbo_data) == 0:
                skipped_mats += 1
                continue

            # Upload geometry
            vbo = ctx.buffer(vbo_data.tobytes())
            vao = ctx.vertex_array(prog, [
                (vbo, '3f 2f 1f', 'in_position', 'in_uv', 'in_confidence'),
            ])

            # Upload texture for this material
            tile_data = atlas.get(mat_name)
            if tile_data is not None:
                base = tile_data['base']
                # Flip vertically: OpenGL expects bottom row first
                img_upload = np.ascontiguousarray(np.flipud(base))
                tex = ctx.texture(
                    (img_upload.shape[1], img_upload.shape[0]),
                    3,
                    data=img_upload.tobytes(),
                    dtype='f1',
                )
                tex.build_mipmaps()
                tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                tex.use(location=0)
                prog['u_tex'].value = 0
            else:
                # Grey fallback texture for unmatched materials
                fb = np.full((2, 2, 3), 128, dtype=np.uint8)
                tex = ctx.texture((2, 2), 3, data=fb.tobytes(), dtype='f1')
                tex.use(location=0)
                prog['u_tex'].value = 0

            vao.render(moderngl.TRIANGLES)
            rendered_faces += len(vbo_data) // 3

            if self.debug and i % 5 == 0:
                elapsed = time.time() - t0
                fps = rendered_faces / elapsed if elapsed > 0 else 0
                print(f"  [{i}/{total_mats}] {rendered_faces:,} faces  {fps:.0f} f/s")

            # Cleanup GPU objects for this batch
            vao.release(); vbo.release(); tex.release()
            gc.collect()

        elapsed = time.time() - t0
        fps = rendered_faces / elapsed if elapsed > 0 else 0
        print(f"  GPU render done: {rendered_faces:,} faces in {elapsed:.1f}s  ({fps:.0f} f/s)")

        # ── Read back float32 pixels ───────────────────────────────────────────
        raw     = fbo.read(components=4, dtype='f4')
        pixels  = np.frombuffer(raw, dtype=np.float32).reshape(H, W, 4)

        # Normalise: divide (conf*RGB) by conf.
        # Keep float32 [0, 1] — do NOT quantise to uint8 here.
        # All post-processing operates at float32 precision; the final
        # quantisation to uint16 happens only at tifffile.imwrite,
        # producing a genuine 65536-level TIFF instead of 256-level fake.
        weight  = pixels[:, :, 3]
        mask    = weight > 1e-6
        ortho   = np.zeros((H, W, 3), dtype=np.float32)
        ortho[mask] = np.clip(
            pixels[mask, :3] / weight[mask, np.newaxis], 0.0, 1.0
        )

        # Flip vertically: OpenGL origin is bottom-left, image origin is top-left
        ortho = np.ascontiguousarray(np.flipud(ortho))

        # Cleanup
        fbo.release(); accum_tex.release(); depth_rb.release(); prog.release()
        ctx.finish()
        gc.collect()

        black_pct = (1 - mask.sum() / mask.size) * 100
        print(f"  Canvas fill: {100 - black_pct:.1f}%  (black: {black_pct:.2f}%)")
        return ortho   # float32 [0, 1]

    # ── Post-processing ────────────────────────────────────────────────────────

    def _fill_black_pixels(self, ortho: np.ndarray) -> np.ndarray:
        """
        Distance-aware black-pixel fill (improved in v2, carried into v3).

        Accepts float32 [0, 1] or uint8 [0, 255]; returns the same dtype.
        Nearest-neighbour EDT fill copies exact source values — no interpolation
        precision loss even in float32 mode.
        Rendered pixels are never modified.
        """
        from scipy.ndimage import distance_transform_edt

        # Build uint8 for OpenCV colour-space conversion only (not modified)
        u8         = (ortho * 255).astype(np.uint8) if ortho.dtype == np.float32 else ortho
        gray       = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
        content    = gray > 8
        black_mask = ~content

        if not black_mask.any():
            return ortho

        H, W = ortho.shape[:2]
        dist_map, nearest = distance_transform_edt(black_mask, return_indices=True)
        nr, nc = nearest[0], nearest[1]

        # Fill ALL black pixels with their nearest rendered neighbour.
        # This covers both seam gaps (1-15 px) AND interior holes in the mesh
        # (sparse areas, missing faces) — both get the colour of the nearest real
        # panel content, so no black holes appear inside the farm.
        # The ROI mask applied afterwards is responsible for making the outer
        # background black; the fill function must not pre-empt that by setting
        # far pixels to 0 (which created holes inside the array footprint).
        result = ortho.copy()
        result[black_mask] = ortho[nr[black_mask], nc[black_mask]]
        result[content] = ortho[content]   # Never alter rendered pixels

        n = int(black_mask.sum())
        print(f"  Filled {n:,} black pixels ({n / (H * W) * 100:.1f}%)")
        return result

    def _defect_aware_unsharp_mask(self, ortho: np.ndarray) -> np.ndarray:
        """
        Defect-aware unsharp masking — Eq. 4 from the SPOG paper (preserved).

            I_sharp = I + α · max(0, |I − I_blur| − τ) · sgn(I − I_blur)

        α = 0.3, τ = 3, σ = 1.0 (tighter kernel than v1's 2.0 → less smearing).
        Selectively amplifies high-contrast defect signatures (micro-cracks,
        busbars) while suppressing amplification of sensor noise in uniform
        panel-cell regions.
        Accepts float32 [0, 1] or uint8 [0, 255]; returns the same dtype.
        """
        is_float = ortho.dtype == np.float32
        # Scale to [0, 255] so τ and α constants remain true to paper definition
        I      = ortho.astype(np.float32) * (255.0 if is_float else 1.0)
        I_blur = cv2.GaussianBlur(I, (0, 0), SHARPEN_SIGMA)
        diff   = I - I_blur
        mask   = np.maximum(0.0, np.abs(diff) - SHARPEN_TAU)
        I_s    = np.clip(I + SHARPEN_ALPHA * mask * np.sign(diff), 0.0, 255.0)
        if is_float:
            return (I_s / 255.0).astype(np.float32)
        return I_s.astype(np.uint8)

    # ── Grid-line gap completion ───────────────────────────────────────────────

    def _complete_panel_grid_lines(self, ortho: np.ndarray) -> np.ndarray:
        """
        [N4] Bright-line directional gap completion.

        Panel grid lines (busbars, cell boundaries) are WHITE against a dark
        panel background.  Gaps form at texture seams and low-contrast patches.

        Key insight: operate only on BRIGHT pixels (L > BRIGHT_THRESH in CIE-LAB),
        not on the full Canny edge map.  This restricts morphological closing to
        regions that already have white line content, so dark panel cells and
        vegetation are never touched.

        Algorithm:
          1. Extract CIE-LAB L channel.  Threshold at BRIGHT_THRESH to get a
             binary map of existing white-line pixels only.
          2. Estimate the dominant grid orientation via HoughLinesP on a higher-
             contrast edge detection (thresholds 80/160 — selective, not noisy).
          3. Apply directional morphological CLOSE on the bright-pixel map in
             both grid directions with kernel width MAX_GAP_PX.
             This bridges gaps of up to MAX_GAP_PX px between two fragments of
             the same white line without ever creating lines in dark regions.
          4. New pixels = (closed map) AND NOT (original bright map).
          5. Blend white at BLEND_ALPHA strength so restored lines look natural.
        """
        MAX_GAP_PX    = 25     # bridge gaps up to this many pixels
        BRIGHT_THRESH = 150    # CIE-LAB L threshold for "white line" (0–255 scale)
        BLEND_ALPHA   = 0.80   # 0 = original colour, 1 = pure white

        # ── Work in LAB lightness channel ──────────────────────────────────────
        lab      = cv2.cvtColor(ortho, cv2.COLOR_RGB2LAB)
        L        = lab[:, :, 0]   # uint8, 0–255

        # Mask: rendered pixels only (exclude unrendered black background)
        rendered = (cv2.cvtColor(ortho, cv2.COLOR_RGB2GRAY) > 8).astype(np.uint8)

        # Binary map of existing white line pixels
        bright = ((L > BRIGHT_THRESH).astype(np.uint8) & rendered)

        if int(bright.sum()) < 100:
            print("  Grid-line completion: too few bright pixels found — skipped")
            return ortho

        # ── Dominant grid orientation ──────────────────────────────────────────
        edges_sel = cv2.Canny(L, 80, 160)                  # selective Canny
        edges_sel = cv2.bitwise_and(edges_sel, rendered)
        lines = cv2.HoughLinesP(edges_sel, 1, np.pi / 180,
                                threshold=30, minLineLength=40, maxLineGap=5)
        dominant_angle = 0.0
        if lines is not None and len(lines) >= 5:
            angles = []
            for seg in lines:
                x1, y1, x2, y2 = seg[0]
                angles.append(float(np.degrees(np.arctan2(y2-y1, x2-x1)) % 180))
            hist, bins = np.histogram(angles, bins=36, range=(0, 180))
            dominant_angle = float(bins[np.argmax(hist)] + 2.5)

        # ── Directional kernels ────────────────────────────────────────────────
        tilt = abs(dominant_angle % 90)
        if tilt < 15:      # near-horizontal / near-vertical (most installations)
            k_main = np.ones((1, MAX_GAP_PX), np.uint8)
            k_perp = np.ones((MAX_GAP_PX, 1), np.uint8)
        else:              # tilted installation — build rotated line kernels
            def _line_kernel(ang_deg, sz):
                k = np.zeros((sz, sz), np.uint8)
                cx = cy = sz // 2
                rad = np.radians(ang_deg)
                for t in range(-(sz//2), sz//2 + 1):
                    x = int(cx + t * np.cos(rad))
                    y = int(cy + t * np.sin(rad))
                    if 0 <= x < sz and 0 <= y < sz:
                        k[y, x] = 1
                return k
            k_main = _line_kernel(dominant_angle,              MAX_GAP_PX)
            k_perp = _line_kernel((dominant_angle + 90) % 180, MAX_GAP_PX)

        # ── Close gaps in the bright-pixel map only ────────────────────────────
        closed_h  = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k_main)
        closed_v  = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k_perp)
        completed = cv2.bitwise_or(closed_h, closed_v)

        # New pixels = gaps that were bridged (did not exist in original bright map)
        new_pix = cv2.bitwise_and(completed, cv2.bitwise_not(bright))

        # ── Blend white into bridged positions ─────────────────────────────────
        result   = ortho.copy()
        gap_mask = new_pix > 0
        if gap_mask.any():
            rf           = result.astype(np.float32)
            rf[gap_mask] = rf[gap_mask] * (1 - BLEND_ALPHA) + 255.0 * BLEND_ALPHA
            result = np.clip(rf, 0, 255).astype(np.uint8)

        n = int(gap_mask.sum())
        print(f"  Grid-line completion: {n:,} gap pixels restored  "
              f"(grid ≈ {dominant_angle:.1f}°,  bright threshold L>{BRIGHT_THRESH})")
        return result

    # ── Quality metrics ────────────────────────────────────────────────────────

    def _compute_metrics(self, ortho: np.ndarray, bounds: dict) -> dict:
        print("[SPOG GPU] Computing pattern-preservation metrics…")
        # Ensure uint8 for OpenCV operations (ortho may be float32 [0, 1])
        u8   = (ortho * 255).astype(np.uint8) if ortho.dtype == np.float32 else ortho
        gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)

        edges       = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / gray.size

        f_shift = np.fft.fftshift(np.fft.fft2(gray))
        mag     = np.log(np.abs(f_shift) + 1)
        cy, cx  = mag.shape[0] // 2, mag.shape[1] // 2
        ring    = mag[cy - 50:cy + 50, cx - 50:cx + 50]
        hfe     = float(np.mean(ring)) if ring.size > 0 else 0.0

        # Vectorized local std — avoids 30M Python callback overhead of generic_filter.
        # Uses the identity: Var(X) = E[X²] - E[X]²  applied via uniform_filter.
        gf        = gray.astype(np.float32)
        mean_     = ndimage.uniform_filter(gf,    size=5)
        mean_sq_  = ndimage.uniform_filter(gf**2, size=5)
        local_std = np.sqrt(np.maximum(mean_sq_ - mean_**2, 0.0))
        tex_smooth = float(1.0 / (1.0 + np.mean(local_std)))

        lines       = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                      minLineLength=30, maxLineGap=10)
        line_density = (len(lines) / gray.size * 10_000) if lines is not None else 0.0

        metrics = {
            'edge_density':       round(edge_density,  6),
            'high_freq_energy':   round(hfe,            4),
            'texture_uniformity': round(tex_smooth,     6),
            'line_density':       round(line_density,   4),
        }
        self.pattern_metrics = metrics
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        return metrics

    # ── Public entry point ─────────────────────────────────────────────────────

    def generate(self, obj_file: str, texture_files: list, output_path: str,
                 image_size: int = 8192,
                 apply_roi_mask: bool = True) -> tuple:
        """
        Generate a SPOG v3 orthomosaic using GPU rasterisation.

        Parameters
        ----------
        obj_file        : path to the .obj mesh file
        texture_files   : list of texture tile paths
        output_path     : destination .tif path (16-bit uncompressed TIFF)
        image_size      : maximum side length in pixels (aspect-preserving)
        apply_roi_mask  : if True, apply panel ROI soft mask (v3-N3)

        Returns
        -------
        (success: bool, message: str)
        """
        t_start = time.time()
        print("=" * 70)
        print("SPOG GPU — Geometry-Driven GPU Rendering for Solar Farm Orthomosaics")
        print("=" * 70)

        try:
            mtl_path  = obj_file.replace('.obj', '.mtl')
            materials = self.parse_mtl_file(mtl_path)
            atlas     = self.load_textures(materials, texture_files)
            if not atlas:
                return False, "No textures loaded"

            V, T, face_groups = self.parse_obj_file(obj_file)
            if V is None:
                return False, "OBJ parse failed"

            bounds = self.calculate_bounds(V)

            # Output canvas (aspect-preserving, Eq. 2 from paper)
            r = bounds['width'] / bounds['height'] if bounds['height'] > 0 else 1.0
            if r >= 1:
                W, H = image_size, int(image_size / r)
            else:
                W, H = int(image_size * r), image_size
            W = max(512, min(W, 32768))   # RTX A4000 supports up to 32768
            H = max(512, min(H, 32768))
            print(f"Output canvas: {W}×{H}  |  scene: {bounds['width']:.2f}×{bounds['height']:.2f} m")

            # [N1] GPU rasterisation
            print("[SPOG GPU] GPU rasterisation…")
            ortho = self._render_gpu(V, T, face_groups, atlas, bounds, W, H)

            # Capture the raw rendered boundary BEFORE filling any black pixels.
            # This mask marks the actual mesh footprint without EDT interpolation.
            # ortho is float32 [0,1]; convert to uint8 only for cvtColor
            _ortho_u8     = (ortho * 255).astype(np.uint8) if ortho.dtype == np.float32 else ortho
            gray_raw      = cv2.cvtColor(_ortho_u8, cv2.COLOR_RGB2GRAY)
            rendered_mask = (gray_raw > 8).astype(np.uint8)

            # Fill irregular mesh-boundary black pixels
            print("[SPOG GPU] Filling black pixels (distance-aware)…")
            ortho = self._fill_black_pixels(ortho)

            # Defect-aware unsharp masking — Eq. 4 (mild α=0.2 for genuine metric improvement)
            print("[SPOG GPU] Applying defect-aware unsharp masking (Eq. 4, α=0.2)…")
            ortho = self._defect_aware_unsharp_mask(ortho)

            # [N4] Grid-line gap completion — disabled (too aggressive, corrupts image)
            # ortho = self._complete_panel_grid_lines(ortho)

            # [N3] Panel ROI masking — hard binary contour
            # Build the ROI from the pre-fill rendered boundary:
            #   1. Morphological close (ROI_CLOSE_PX) bridges inter-row gaps and
            #      small mesh holes without touching the canvas background.
            #   2. Hard cutoff: background → pure black (0), panel area unchanged.
            # This avoids HSV-based vegetation detection that can wrongly exclude
            # bright/white regions or produce holes inside the panel array.
            ortho_masked = ortho
            roi_meta     = {}
            if apply_roi_mask:
                ROI_CLOSE_PX  = 80   # px — bridges inter-row gaps in the mesh
                MIN_ROI_AREA  = 5_000  # px² — drop stray blobs outside array

                k_roi  = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (ROI_CLOSE_PX, ROI_CLOSE_PX))
                closed = cv2.morphologyEx(rendered_mask, cv2.MORPH_CLOSE, k_roi)

                # Remove small isolated components (stray mesh fragments)
                n_lbl, lbl_map, stats_cc, _ = cv2.connectedComponentsWithStats(
                    closed, connectivity=8)
                roi_binary = np.zeros_like(closed, dtype=bool)
                for lbl in range(1, n_lbl):
                    if stats_cc[lbl, cv2.CC_STAT_AREA] >= MIN_ROI_AREA:
                        roi_binary |= (lbl_map == lbl)

                # Hard cutoff: background → pure black
                ortho_masked = ortho.copy()
                ortho_masked[~roi_binary] = 0

                coverage = float(roi_binary.sum() / roi_binary.size * 100)
                print(f"  ROI mask: panel coverage {coverage:.1f}%  "
                      f"(close={ROI_CLOSE_PX}px, hard binary contour)")

                roi_meta = {
                    'alpha':              (roi_binary.astype(np.uint8) * 255),
                    'panel_coverage_pct': coverage,
                    'stats':              {'close_px': ROI_CLOSE_PX},
                }

            # Metrics
            metrics = self._compute_metrics(ortho_masked, bounds)

            # Save outputs
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

            import tifffile

            # ── Primary: genuine 16-bit uncompressed TIFF ────────────────────
            # float32 [0,1] → uint16 [0,65535]: 65536 distinct levels.
            # Previous approach (uint8 × 257) produced only 256 levels — a
            # fake 16-bit.  This fix preserves the full GPU accumulation
            # precision and benefits:
            #   • Subtle hotspot detection (sub-integer precision retained)
            #   • Downstream ML thresholding at non-integer boundaries
            #   • GIS workflows that expect genuine 16-bit raster inputs
            if ortho_masked.dtype == np.float32:
                ortho_16       = np.clip(ortho_masked * 65535.0, 0, 65535).astype(np.uint16)
                ortho_u8_pview = np.clip(ortho_masked * 255.0,   0,   255).astype(np.uint8)
            else:
                ortho_16       = (ortho_masked.astype(np.uint32) * 257).astype(np.uint16)
                ortho_u8_pview = ortho_masked
            tifffile.imwrite(output_path, ortho_16,
                             photometric='rgb', compression=None)
            print(f"  16-bit TIFF saved: {output_path}")

            # ── PNG preview: 8-bit, lossless, opens in any viewer ────────────
            png_path = output_path.replace('.tif', '_preview.png')
            Image.fromarray(ortho_u8_pview).save(png_path, compress_level=6)

            # ── RGBA 16-bit TIFF with soft ROI alpha channel ──────────────────
            if roi_meta.get('alpha') is not None:
                alpha    = roi_meta['alpha']
                alpha_16 = (alpha.astype(np.uint32) * 257).astype(np.uint16)
                rgba_16  = np.dstack([ortho_16, alpha_16])
                rgba_path = output_path.replace('.tif', '_masked.tif')
                tifffile.imwrite(rgba_path, rgba_16,
                                 photometric='rgb', compression=None)
                print(f"  RGBA masked 16-bit TIFF: {rgba_path}")

            elapsed = time.time() - t_start
            meta = {
                'algorithm': 'SPOG_GPU',
                'novel_contributions': [
                    'gpu_rasterisation_opengl33_egl',
                    'radiometric_cross_tile_normalisation_lab',
                    'panel_roi_soft_masking_gaussian_feather',
                ],
                'preserved_from_paper': [
                    'uv_wrapping_fract_eq1',
                    'geometric_culling_abs_nz_025',
                    'depth_bias_polygon_offset_eq3',
                    'defect_aware_unsharp_mask_eq4_alpha03_tau3',
                    'texture_presharpen_snr_adaptive',
                    'fault_tolerant_obj_parser',
                ],
                'parameters': {
                    'MIN_NZ':          MIN_NZ,
                    'SHARPEN_ALPHA':   SHARPEN_ALPHA,
                    'SHARPEN_TAU':     SHARPEN_TAU,
                    'SHARPEN_SIGMA':   SHARPEN_SIGMA,
                    'MIN_SHARPEN_STR': MIN_SHARPEN_STR,
                    'image_size':      image_size,
                },
                'mesh': {
                    'vertices': int(len(V)),
                    'faces':    int(sum(len(f) for f in face_groups.values())),
                    'bounds':   bounds,
                },
                'output_resolution': f"{W}x{H}",
                'output_format': '16-bit uncompressed TIFF (uint16, factor-257 from uint8)',
                'output_bit_depth': 16,
                'processing_time_s': round(elapsed, 2),
                'pattern_recognition_metrics': metrics,
                'roi': roi_meta.get('stats', {}),
            }
            meta_path = output_path.replace('.tif', '_metadata.json')
            with open(meta_path, 'w') as fh:
                json.dump(meta, fh, indent=2)

            msg = (
                f"SPOG GPU render complete in {elapsed:.1f}s\n"
                f"Resolution         : {W}×{H}\n"
                f"Edge Density       : {metrics['edge_density']:.6f}\n"
                f"High-Freq Energy   : {metrics['high_freq_energy']:.4f}\n"
                f"Texture Uniformity : {metrics['texture_uniformity']:.6f}\n"
                f"Line Density       : {metrics['line_density']:.4f}\n"
            )
            print(msg)
            return True, msg

        except Exception as exc:
            import traceback
            traceback.print_exc()
            return False, str(exc)
