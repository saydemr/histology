"""
virtual_he.py
=============

Translate two-channel light-sheet fluorescence microscopy images
(nuclear stain + structural/cytoplasmic stain) into virtual (false-colored)
H&E-like images, for histopathological visualization when no paired
ground-truth H&E is available.

Usage
-----
    python virtual_he.py --c1 path/to/C01_nuclear.tif \
                          --c2 path/to/C02_structural.tif \
                          --out virtual_he.png

Or import and call `run_pipeline(c1_path, c2_path)` from your own code.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import color, exposure, filters, img_as_float32, io, morphology
from skimage.restoration import denoise_nl_means, estimate_sigma


@dataclasses.dataclass
class VirtualHEConfig:
    # preprocessing
    lower_percentile: float = 0.5  # for robust intensity normalization
    upper_percentile: float = 99.7
    rolling_ball_radius: int = 50  # background-subtraction structuring elem radius (px)
    denoise: bool = True
    denoise_patch_size: int = 5
    denoise_patch_distance: int = 6
    empty_channel_snr_multiplier: float = 8.0

    # virtual staining
    hematoxylin_gain: float = 1.0  # overall nuclear stain strength
    eosin_gain: float = 0.75  # overall structural stain strength
    hematoxylin_gamma: float = 0.85
    eosin_gamma: float = 1.0
    saturation_k: float = 6.0  # sigmoid steepness for stain-saturation kinetics

    max_optical_density: float = 0.42  # peak per-stain optical density
    hematoxylin_od_cap: float = 0.30  # nuclei: kept lower than max_optical_density so they read as saturated blue/purple rather than near-black

    # postprocessing
    background_percentile: float = 99.0  # pixels above this treated as background
    clahe_clip_limit: float = 0.01
    clahe_kernel_frac: float = 1 / 16  # CLAHE tile size as a fraction of image size
    unsharp_radius: float = 1.5
    unsharp_amount: float = 0.4

    #  misc
    auto_detect_channel_roles: bool = True
    seed: int = 0


def load_channel(path: str) -> np.ndarray:
    """Load a single-channel microscopy image (TIFF or PNG/etc.) as float32
    in [0, 1] using its own native bit depth range for normalization."""
    path = str(path)
    arr = tifffile.imread(path)

    arr = np.asarray(arr)
    if arr.ndim == 3:
        # If accidentally given an RGB/RGBA PNG, collapse to luminance.
        arr = color.rgb2gray(arr[..., :3])
        return arr.astype(np.float32)

    return arr.astype(np.float32)


# Preprocessing
def subtract_background(img: np.ndarray, radius: int) -> np.ndarray:
    """Morphological white top-hat background subtraction (rolling-ball
    approximation). Removes smooth out-of-focus glow / light-sheet haze
    while preserving sharp nuclear/fiber structures.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    background = cv2.morphologyEx(img.astype(np.float32), cv2.MORPH_OPEN, kernel)
    corrected = img - background
    corrected = np.clip(corrected, 0, None)
    return corrected


def robust_normalize(img: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    """Percentile-based contrast stretch to [0, 1], robust to hot-pixel
    outliers and to varying laser power / exposure across acquisitions."""
    lo, hi = np.percentile(img, [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1e-6
    out = (img - lo) / (hi - lo)
    return np.clip(out, 0, 1)


def estimate_global_range(
    paths: List[str],
    low_pct: float = 0.5,
    high_pct: float = 99.9,
    sample_stride: int = 1,
) -> Tuple[float, float]:
    """Scan a list of raw (un-normalized) channel files and compute a
    dataset-wide (lo, hi) intensity range in raw units.
    """
    if not paths:
        raise ValueError("estimate_global_range: no paths given.")
    sampled = paths[:: max(sample_stride, 1)] or paths[:1]

    all_lo, all_hi = [], []
    for p in sampled:
        arr = load_channel(p)
        lo, hi = np.percentile(arr, [low_pct, high_pct])
        all_lo.append(lo)
        all_hi.append(hi)

    return float(np.median(all_lo)), float(np.max(all_hi))


def _robust_sigma(img: np.ndarray) -> float:
    """Median-absolute-deviation based noise sigma estimate on the finest
    wavelet-free high-frequency residual (img - gaussian-blur). Avoids a
    hard dependency on PyWavelets while remaining robust to outliers."""
    residual = img - ndi.gaussian_filter(img, sigma=1)
    mad = np.median(np.abs(residual - np.median(residual)))
    return float(mad / 0.6745)


def denoise_channel(
    img: np.ndarray, patch_size: int, patch_distance: int
) -> np.ndarray:
    """Light non-local-means denoising."""
    sigma_est = np.mean(estimate_sigma(img))
    h = 0.8 * sigma_est if sigma_est > 0 else 0.01
    return denoise_nl_means(
        img,
        h=h,
        fast_mode=True,
        patch_size=patch_size,
        patch_distance=patch_distance,
        preserve_range=True,
    )


def channel_has_signal(
    raw: np.ndarray, low_pct: float, high_pct: float, snr_multiplier: float = 8.0
) -> bool:
    """Decide whether a raw channel contains real spatial signal, or is
    just sensor/read noise on an effectively blank frame."""
    own_spread = float(np.percentile(raw, high_pct) - np.percentile(raw, low_pct))
    sigma = _robust_sigma(raw)
    if sigma <= 1e-6:
        return (
            own_spread > 1e-6
        )  # perfectly flat noise-free frame is the only edge case here
    return own_spread >= snr_multiplier * sigma


def preprocess_channel(
    raw: np.ndarray,
    cfg: "VirtualHEConfig",
    global_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, bool]:
    """Normalize + background-subtract + denoise one channel."""
    raw = raw.astype(np.float32)

    has_signal = channel_has_signal(
        raw,
        cfg.lower_percentile,
        cfg.upper_percentile,
        cfg.empty_channel_snr_multiplier,
    )
    if not has_signal:
        return np.zeros(raw.shape, dtype=np.float32), True

    if global_range is not None:
        lo, hi = global_range
        img = np.clip((raw - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    else:
        lo, hi = np.percentile(raw, [cfg.lower_percentile, cfg.upper_percentile])
        img = np.clip((raw - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)

    img = subtract_background(img, cfg.rolling_ball_radius)
    img = np.clip(img, 0, 1)
    if cfg.denoise:
        img = denoise_channel(img, cfg.denoise_patch_size, cfg.denoise_patch_distance)
    return np.clip(img, 0, 1).astype(np.float32), False


def _blob_score(img: np.ndarray) -> float:
    """Higher score => image looks more like discrete round nuclear blobs.
    Lower score => image looks like diffuse fibrous/continuous structural signal."""
    thresh = filters.threshold_otsu(img) if img.max() > img.min() else 0.5
    binary = img > thresh
    binary = morphology.remove_small_objects(binary, max_size=3)
    labeled, n_labels = ndi.label(binary)
    if n_labels == 0:
        return 0.0

    sizes = ndi.sum(binary, labeled, range(1, n_labels + 1))
    total_area = binary.size

    mean_size = np.mean(sizes) if len(sizes) else 1.0
    density = n_labels / total_area * 1e4

    score = density / (mean_size**0.5 + 1e-3)
    return float(score)


def detect_nuclear_channel(
    chan_a: np.ndarray, chan_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Returns (nuclear_channel, structural_channel, swapped_flag).
    swapped_flag=True means chan_a was actually the structural channel and
    chan_b was the nuclear one"""
    score_a = _blob_score(chan_a)
    score_b = _blob_score(chan_b)
    if score_b > score_a:
        return chan_b, chan_a, True
    return chan_a, chan_b, False


# Virtual staining: forward Beer-Lambert H&E compositing


def _sigmoid_saturate(x: np.ndarray, k: float) -> np.ndarray:
    """Maps [0,1] concentration proxy through a logistic curve centered at
    0.5, approximating real stain-binding saturation kinetics."""
    return 1.0 / (1.0 + np.exp(-k * (x - 0.5)))


def channels_to_concentrations(
    nuclear: np.ndarray,
    structural: np.ndarray,
    cfg: VirtualHEConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert normalized fluorescence intensities into virtual stain
    'concentration' maps (unitless optical-density proxies) for hematoxylin
    (c_H) and eosin (c_E).
    """
    h_lin = np.clip(nuclear, 0, 1) ** cfg.hematoxylin_gamma
    e_lin = np.clip(structural, 0, 1) ** cfg.eosin_gamma

    h_sat = _sigmoid_saturate(h_lin, cfg.saturation_k)
    e_sat = _sigmoid_saturate(e_lin, cfg.saturation_k)
    lo = _sigmoid_saturate(np.array(0.0), cfg.saturation_k)
    hi = _sigmoid_saturate(np.array(1.0), cfg.saturation_k)
    h_sat = np.clip((h_sat - lo) / (hi - lo + 1e-8), 0, 1)
    e_sat = np.clip((e_sat - lo) / (hi - lo + 1e-8), 0, 1)

    # Nuclear dominance: suppress eosin under strong nuclear signal.
    eosin_suppression = 1.0 - 0.75 * h_sat
    e_sat = e_sat * np.clip(eosin_suppression, 0.15, 1.0)

    c_h = h_sat * cfg.hematoxylin_gain * cfg.hematoxylin_od_cap
    c_e = e_sat * cfg.eosin_gain * cfg.max_optical_density
    return c_h.astype(np.float32), c_e.astype(np.float32)


def beer_lambert_composite(c_h: np.ndarray, c_e: np.ndarray) -> np.ndarray:
    """Forward Beer-Lambert absorbance compositing using the standard
    Ruifrok & Johnston H&E stain vectors. Produces an RGB image in [0, 1]."""
    h, w = c_h.shape
    hed = np.zeros((h, w, 3), dtype=np.float32)
    hed[..., 0] = c_h  # Hematoxylin channel
    hed[..., 1] = c_e  # Eosin channel
    hed[..., 2] = 0.0  # DAB channel unused
    rgb = color.hed2rgb(hed)
    return np.clip(rgb, 0, 1)


# Postprocessing


def white_balance_background(
    rgb: np.ndarray, c_h: np.ndarray, c_e: np.ndarray, cfg: VirtualHEConfig
) -> np.ndarray:
    """Slide background (acellular, low signal in both stains) should read
    as pale pink/white."""
    combined_signal = c_h + c_e
    bg_thresh = np.percentile(combined_signal, 100 - cfg.background_percentile)
    bg_mask = combined_signal <= max(bg_thresh, 1e-4)

    if bg_mask.sum() < 50:  # not enough background pixels found skip
        return rgb

    target_bg = np.array(
        [0.94, 0.88, 0.93]
    )  # pale pink-white, typical clean H&E slide bg
    current_bg = rgb[bg_mask].mean(axis=0)
    current_bg = np.clip(current_bg, 1e-3, 1.0)

    gain = target_bg / current_bg
    gain = np.clip(gain, 0.5, 2.0)  # avoid extreme corrections
    balanced = rgb * gain[None, None, :]
    return np.clip(balanced, 0, 1)


def local_contrast_enhance(rgb: np.ndarray, cfg: VirtualHEConfig) -> np.ndarray:
    """CLAHE on the L channel of L*a*b* only, to avoid perturbing hue/chroma."""
    lab = color.rgb2lab(rgb)
    L = lab[..., 0] / 100.0
    kernel_size = max(8, int(min(rgb.shape[:2]) * cfg.clahe_kernel_frac))
    L_eq = exposure.equalize_adapthist(
        L, kernel_size=kernel_size, clip_limit=cfg.clahe_clip_limit
    )
    lab[..., 0] = L_eq * 100.0
    out = color.lab2rgb(lab)
    return np.clip(out, 0, 1)


def unsharp(rgb: np.ndarray, radius: float, amount: float) -> np.ndarray:
    """Simple, explicit unsharp mask implemented directly with
    scipy.ndimage.gaussian_filter."""
    blurred = ndi.gaussian_filter(rgb, sigma=(radius, radius, 0))
    sharpened = rgb + amount * (rgb - blurred)
    return np.clip(sharpened, 0, 1)


def reinhard_color_transfer(rgb: np.ndarray, ref_rgb: np.ndarray) -> np.ndarray:
    """match mean/std of L*a*b* channels to a real reference H&E
    patch, if one is available. Purely a postprocessing grading step; not
    required for the pipeline to work."""
    src_lab = color.rgb2lab(rgb)
    ref_lab = color.rgb2lab(ref_rgb)
    out_lab = np.empty_like(src_lab)
    for i in range(3):
        s_mean, s_std = src_lab[..., i].mean(), src_lab[..., i].std() + 1e-6
        r_mean, r_std = ref_lab[..., i].mean(), ref_lab[..., i].std()
        out_lab[..., i] = (src_lab[..., i] - s_mean) * (r_std / s_std) + r_mean
    out_lab[..., 0] = np.clip(out_lab[..., 0], 0, 100)
    return np.clip(color.lab2rgb(out_lab), 0, 1)


# Full pipeline


def run_pipeline(
    c1_path: str,
    c2_path: str,
    cfg: Optional[VirtualHEConfig] = None,
    reference_he_path: Optional[str] = None,
    c1_global_range: Optional[Tuple[float, float]] = None,
    c2_global_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """End-to-end: load two fluorescence channels -> virtual H&E RGB image
    in [0, 1] float32, shape (H, W, 3)."""
    cfg = cfg or VirtualHEConfig()

    raw_a = load_channel(c1_path)
    raw_b = load_channel(c2_path)

    chan_a, empty_a = preprocess_channel(raw_a, cfg, global_range=c1_global_range)
    chan_b, empty_b = preprocess_channel(raw_b, cfg, global_range=c2_global_range)

    if empty_a and empty_b:
        # No real signal in either channel -- render plain background
        # instead of running staining/CLAHE on noise.
        print(
            "[virtual_he] Both channels detected as empty (no signal above "
            "noise floor) -- rendering plain background, skipping staining."
        )
        c_h = np.zeros_like(chan_a)
        c_e = np.zeros_like(chan_a)
        rgb = beer_lambert_composite(c_h, c_e)
        rgb = white_balance_background(rgb, c_h, c_e, cfg)
        return rgb.astype(np.float32)

    if cfg.auto_detect_channel_roles:
        nuclear, structural, swapped = detect_nuclear_channel(chan_a, chan_b)
        if swapped:
            print(
                "[virtual_he] Auto-detection: input channel order looked "
                "swapped relative to (C1=nuclear, C2=structural); "
                "correcting automatically based on blob statistics."
            )
    else:
        nuclear, structural = chan_a, chan_b

    c_h, c_e = channels_to_concentrations(nuclear, structural, cfg)
    rgb = beer_lambert_composite(c_h, c_e)

    rgb = white_balance_background(rgb, c_h, c_e, cfg)
    rgb = local_contrast_enhance(rgb, cfg)
    rgb = unsharp(rgb, cfg.unsharp_radius, cfg.unsharp_amount)

    if reference_he_path is not None:
        ref = img_as_float32(io.imread(reference_he_path)[..., :3])
        rgb = reinhard_color_transfer(rgb, ref)

    return rgb.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Virtual H&E staining for two-channel light-sheet microscopy."
    )
    parser.add_argument(
        "--c1", required=True, help="Path to channel 1 image (TIFF/PNG)."
    )
    parser.add_argument(
        "--c2", required=True, help="Path to channel 2 image (TIFF/PNG)."
    )
    parser.add_argument("--out", help="Output image path.")
    parser.add_argument(
        "--reference-he",
        default=None,
        help="Optional reference real H&E patch for color grading.",
    )
    parser.add_argument(
        "--no-auto-detect",
        action="store_true",
        help="Disable automatic nuclear/structural channel detection.",
    )
    parser.add_argument(
        "--no-denoise", action="store_true", help="Disable NLM denoising step."
    )
    args = parser.parse_args()

    cfg = VirtualHEConfig(
        auto_detect_channel_roles=not args.no_auto_detect,
        denoise=not args.no_denoise,
    )

    rgb = run_pipeline(args.c1, args.c2, cfg, reference_he_path=args.reference_he)

    if args.out is None:
        out_path = (
            Path(args.c1).parent.parent
            / "out"
            / f"slice_{Path(args.c1).stem.split('_')[1]}.png"
        )
    else:
        out_path = Path(args.out)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(str(out_path), (rgb * 255).astype(np.uint8))
    print(f"[virtual_he] Saved virtual H&E image to {out_path.resolve()}")


if __name__ == "__main__":
    main()
