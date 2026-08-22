# Virtual H&E Staining of Two-Channel Light-Sheet Microscopy

## Scripts
- `main.py` — main pipeline (run this). No training data or GPU required.
- `virtual_he.py` — the virtual H&E algorithm itself, with tunable parameters in the `VirtualHEConfig` dataclass at the top.

## Methodology and Justification

There is no paired ground-truth H&E available, which rules out supervised pix2pix-style training outright, and makes unpaired/unsupervised GAN training (CycleGAN) a poor fit for a first deliverable: it needs a real reference H&E dataset (not provided), non-trivial GPU training time, and offers no guarantee that nuclear morphology is preserved exactly (cycle- consistency loss *encourages* but does not *enforce* pixel-level structural fidelity) — a problem given that "preserve morphological accuracy" is an explicit objective.

Instead, `virtual_he.py` implements a **physics-based classical mapping**: a forward run of the same **Beer-Lambert absorbance model that real H&E staining follows**, using the standard Ruifrok & Johnston stain vectors (the same matrix `skimage.color.hed2rgb`/`rgb2hed` use for real histology color deconvolution — here it's run forward instead of in reverse).
Concretely:

    I_rgb = I0 * 10^(-A_H * c_H - A_E * c_E)

where `c_H` and `c_E` are "virtual stain concentration" maps derived from the two fluorescence channels, and `A_H`, `A_E` are the fixed hematoxylin / eosin absorption vectors.

This is the standard approach in the virtual-histology literature for turning fluorescence channels (light-sheet, MUSE, etc.) into pseudo-H&E for visualization, and was chosen over a generative model here because it:
- Needs no training data or GPU — runs immediately on your data.
- Is fully deterministic and reproducible (verified below).
- Produces colors that mix the way real absorptive stains do, so hue behavior (purple nuclei vs. pink stroma, and how they blend) matches what a pathologist expects, rather than an arbitrary color LUT.
- Is a **pointwise/local pixel mapping**, not a generative hallucination — it cannot invent or move structures, only recolor what's there. This is what actually gives the morphology-preservation guarantee the task asks for; a GAN cannot make that guarantee.


## Preprocessing

1. **Robust percentile normalization** (0.5–99.7 percentile by default) per channel, rather than min/max — light-sheet data commonly has isolated hot pixels and saturation artifacts that would otherwise dominate a min/max stretch.
2. **Background subtraction** via morphological opening (a rolling-ball style white top-hat), removing smooth out-of-focus glow / light-sheet scattering haze while preserving sharp nuclear and fiber edges.
3. **Denoising**: light non-local-means (NLM) denoising, chosen over Gaussian blur because it smooths flat/noisy background regions while better preserving thin fiber and nuclear-boundary edges — important since morphology preservation is an explicit goal. Falls back to a MAD-based noise-sigma estimate if PyWavelets isn't installed (avoids a hard dependency).
4. **Automatic channel-role detection**: rather than trusting that "channel 1 = nuclear," each preprocessed channel is scored by a blob-density / compactness statistic (small, numerous, high-contrast connected components → high score → nuclear-like). Whichever channel scores higher is treated as the nuclear channel. This makes the pipeline robust to channel-order mistakes in file naming or acquisition metadata.

## Model / algorithm (virtual staining)

1. Nuclear channel intensity → hematoxylin concentration map `c_H`. Structural channel intensity → eosin concentration map `c_E`.
2. Each is passed through a **logistic ("sigmoid") saturation curve** rather than kept linear, approximating real stain-binding saturation kinetics (binding sites saturate; stains are not linear absorbers at all concentrations). Critically, the logistic output is *not* re-stretched to fill [0,1] afterward — doing so would re-inflate background signal and was an early bug that made nuclei render as solid black (see note below).
3. **Nuclear dominance**: eosin concentration is multiplicatively suppressed wherever hematoxylin signal is strong, mirroring the real biology (densely packed chromatin excludes most eosin-binding material), which is what lets nuclei render as blue/purple against a pink background instead of black holes.
4. The two concentration maps are composited into RGB via `skimage.color.hed2rgb` (forward Beer-Lambert, base-10 convention). The per-stain optical-density ceiling is deliberately kept well under 1.0 (default 0.42 for eosin, 0.30 for hematoxylin) — `hed2rgb`'s base-10 convention saturates to near-black quickly, above roughly OD 0.5–0.6 for a single stain, which was empirically verified.

## Postprocessing

1. **Background white-balancing**: pixels with low combined optical density (acellular slide background) are used to estimate the current background color, which is then gain-corrected toward a pale pink-white target (matching how clean H&E slide background actually looks), with gain clipped to [0.5, 2.0] to avoid overcorrection artifacts.
2. **CLAHE** (contrast-limited adaptive histogram equalization) applied only to the L channel of L\*a\*b\* space, boosting local contrast without perturbing hue/chroma balance (which would otherwise shift the H&E purple/pink balance).
3. **Unsharp masking** for print-like crispness — implemented as an explicit Gaussian-based formula (`sharpened = image + amount*(image - blurred)`) rather than via `skimage.filters.unsharp_mask`, for a determinism reason noted below.
4. **Optional Reinhard-style color transfer** (mean/std matching in L\*a\*b\*) against a real reference H&E patch, if one is supplied via `--reference-he`.


## Usage (single pair)

```bash
uv sync

uv run virtual_he.py --c1 path/to/C01.tif --c2 path/to/C02.tif --out virtual_he.png
```

Useful flags:
- `--no-auto-detect` — trust the given channel order as-is (nuclear=c1).
- `--no-denoise` — skip NLM denoising (faster, slightly noisier).
- `--reference-he path/to/real_he_patch.png` — color-grade the output to
  match a real reference H&E image via Reinhard transfer.

All tunable parameters (gains, gammas, OD ceilings, CLAHE strength, etc.) are in the `VirtualHEConfig` dataclass at the top of `virtual_he.py` and are commented with what each one controls.

## Batch processing (255 slice-pairs)

`main.py` runs `virtual_he.py` over a directory laid out as:

    ./data/C01/slice_0000.tif ... slice_0255.tif
    ./data/C02/slice_0000.tif ... slice_0255.tif

```bash
uv run main.py --data-dir ./data --out-dir ./data/virtual_he
```

Pairs are matched by filename (not by assuming a contiguous 0000-0255 range), so gaps or a different total count than expected are handled gracefully with a warning rather than a crash. Useful flags: `--workers N` (default: all CPU cores), `--limit N` (quick test on the first N pairs before committing to a full run), `--pattern` (override the glob if your filenames differ), `--global-range-stride N` (scan every Nth file during the whole-stack normalization pre-pass instead of every file, for a faster but coarser estimate on very large stacks), `--no-global-range` (fall back to independent per-slice normalization -- not recommended, see below).

### Performance

Single-pair time is ~30s end-to-end (measured on a 2048x2048 image, single CPU core), i.e. **~1.5 hours single-threaded for all 255 pairs, dropping roughly linearly with `--workers`** (e.g. ~11 min on 8 cores) since `main.py` parallelizes across pairs with `multiprocessing.Pool` -- each pair is fully independent, so this is a clean embarrassingly-parallel speedup with no correctness risk, unlike squeezing more performance out of any single image's processing.

## Requirement

- [uv](https://docs.astral.sh/uv/) - Astral's command line tool


## Repo Structure
```
├── data
│   ├── C01
│   ├── C02
├── main.py
├── pyproject.toml
├── uv.lock
└── virtual_he.py
```
## Setting up the environment

```bash
uv sync
```

## Usage

```bash
uv run main.py --data-dir ./data
```
or 

```bash
uv run virtual_he.py --c01 ./data/C01 --c02 ./data/C02 --out-dir ./data/out_virtual_he
```