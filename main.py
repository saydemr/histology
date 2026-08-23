"""
Usage
-----
    python main.py --data-dir ./data

Useful flags:
    --workers N       Number of parallel worker processes (default: all cores)
    --limit N         Only process the first N pairs (for a quick test run)
    --pattern GLOB    Override the slice filename pattern (default: slice_*.tif)
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from skimage import io as skio

from virtual_he import VirtualHEConfig, estimate_global_range, run_pipeline


def find_pairs(data_dir: Path, pattern: str) -> list[tuple[Path, Path, str]]:
    """Match slice files in C01/ and C02/ by filename, not by assuming a
    contiguous index range"""
    c01_dir = data_dir / "C01"
    c02_dir = data_dir / "C02"
    if not c01_dir.is_dir() or not c02_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {c01_dir} and {c02_dir} to both exist as directories."
        )

    c01_files = {p.name: p for p in c01_dir.glob(pattern)}
    c02_files = {p.name: p for p in c02_dir.glob(pattern)}

    common = sorted(set(c01_files) & set(c02_files))
    missing_in_c02 = sorted(set(c01_files) - set(c02_files))
    missing_in_c01 = sorted(set(c02_files) - set(c01_files))
    if missing_in_c02:
        print(
            f"[batch] WARNING: {len(missing_in_c02)} file(s) in C01 have no C02 match, skipping: "
            f"{missing_in_c02[:5]}{'...' if len(missing_in_c02) > 5 else ''}"
        )
    if missing_in_c01:
        print(
            f"[batch] WARNING: {len(missing_in_c01)} file(s) in C02 have no C01 match, skipping: "
            f"{missing_in_c01[:5]}{'...' if len(missing_in_c01) > 5 else ''}"
        )

    return [(c01_files[name], c02_files[name], name) for name in common]


def _process_one(args) -> tuple[str, float, bool, str | None]:
    """Runs in a worker process. Returns (filename, elapsed_seconds, was_empty, error_or_None)."""
    c01_path, c02_path, name, out_dir, cfg, c1_range, c2_range = args
    t0 = time.time()

    rgb = run_pipeline(
        str(c01_path),
        str(c02_path),
        cfg,
        c1_global_range=c1_range,
        c2_global_range=c2_range,
    )
    out_path = Path(out_dir) / (Path(name).stem + ".png")
    skio.imsave(
        str(out_path),
        (np.clip(rgb, 0, 1) * 255).astype(np.uint8),
        check_contrast=False,
    )
    was_empty = bool(rgb.std(axis=(0, 1)).max() < 0.01)
    return (name, time.time() - t0, was_empty, None)


def main():
    parser = argparse.ArgumentParser(
        description="Batch virtual H&E staining over paired C01/C02 slice directories."
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Directory containing C01/ and C02/ subfolders.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory.",
    )
    parser.add_argument(
        "--pattern", default="slice_*.tif", help="Glob pattern for slice filenames."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker processes (default: all CPU cores).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N pairs (for a quick test).",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Disable NLM denoising (faster, slightly noisier).",
    )
    parser.add_argument(
        "--no-global-range",
        action="store_true",
        help="Disable the whole-stack normalization pre-pass and fall back to "
        "per-slice percentile normalization (not recommended -- see METHODS.md; "
        "mainly useful if the pre-pass itself is too slow on a very large stack).",
    )
    parser.add_argument(
        "--global-range-stride",
        type=int,
        default=1,
        help="Scan every Nth file when estimating the whole-stack intensity range "
        "(default: 1, scan all files). Increase for a faster but coarser estimate "
        "on very large stacks.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(data_dir, args.pattern)
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        print("[batch] No matching pairs found. Check --data-dir and --pattern.")
        return

    cfg = VirtualHEConfig(denoise=not args.no_denoise)

    c1_range = c2_range = None
    if not args.no_global_range:
        t0 = time.time()
        c1_paths = [str(c01) for c01, _, _ in pairs]
        c2_paths = [str(c02) for _, c02, _ in pairs]
        print(
            f"[batch] Estimating whole-stack intensity range from {len(pairs)} pairs "
            f"(stride={args.global_range_stride})..."
        )
        c1_range = estimate_global_range(
            c1_paths,
            cfg.lower_percentile,
            cfg.upper_percentile,
            args.global_range_stride,
        )
        c2_range = estimate_global_range(
            c2_paths,
            cfg.lower_percentile,
            cfg.upper_percentile,
            args.global_range_stride,
        )
        print(
            f"[batch] C1 range: {c1_range}, C2 range: {c2_range} (raw units, {time.time() - t0:.1f}s)"
        )

    workers = args.workers or cpu_count()
    print(
        f"[batch] {len(pairs)} pairs found. Using {workers} worker process(es). Output -> {out_dir}"
    )

    tasks = [
        (c01, c02, name, out_dir, cfg, c1_range, c2_range) for c01, c02, name in pairs
    ]

    t_start = time.time()
    n_done = 0
    n_errors = 0
    n_empty = 0
    with Pool(processes=workers) as pool:
        for name, elapsed, was_empty, error in pool.imap_unordered(_process_one, tasks):
            n_done += 1
            if error:
                n_errors += 1
                print(
                    f"[batch] ({n_done}/{len(pairs)}) FAILED {name} after {elapsed:.1f}s: {error}"
                )
            else:
                if was_empty:
                    n_empty += 1
                avg = (time.time() - t_start) / n_done
                eta = avg * (len(pairs) - n_done)
                empty_tag = " [empty/background]" if was_empty else ""
                print(
                    f"[batch] ({n_done}/{len(pairs)}) {name} done in {elapsed:.1f}s{empty_tag} "
                    f"| avg {avg:.1f}s/pair | ETA {eta / 60:.1f} min"
                )

    total = time.time() - t_start
    print(
        f"[batch] Finished: {n_done - n_errors}/{len(pairs)} succeeded ({n_empty} detected as "
        f"empty/background), {n_errors} failed, in {total / 60:.1f} min "
        f"({total / max(n_done, 1):.1f}s/pair average, {workers} workers)."
    )


if __name__ == "__main__":
    main()
