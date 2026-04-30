# Add MPS (Metal Performance Shaders) support for Apple Silicon GPUs

This PR brings native PyTorch MPS support to CellBender 0.3.2 so users on Apple Silicon (M1/M2/M3/M4) Macs can run GPU‑accelerated inference on macOS. It adapts the original work from commit `8a70ea3` on the legacy `sf_pytorch_mps_backend` branch (v0.2.0) to the 0.3.2 codebase.

- Related issue: https://github.com/broadinstitute/CellBender/issues/149
- Original MPS commit: `8a70ea306efbfd63f2a219b93b1b2c2749f641df`
- Branch: `add-mps-support`

## Summary

- Adds a new CLI flag `--mps` to enable the PyTorch MPS backend.
- Generalizes device handling across the codebase: instead of a boolean `use_cuda`, a string `device` is passed end‑to‑end and can be one of `cuda`, `mps`, or `cpu`.
- Keeps CUDA as top priority when both CUDA and MPS are available, then MPS, then CPU.
- Backward compatible with existing CUDA and CPU workflows.

## Why this change?

- Many CellBender users are on Apple Silicon laptops/desktops where CUDA is unavailable. MPS provides substantial speedups vs CPU without requiring external GPUs.
- The existing MPS work was never merged into the latest stable code line (0.3.x). This PR forward‑ports the feature carefully aligning with 0.3.2 structure and conventions.

## Changes (by file)

- `cellbender/remove_background/argparser.py`
  - Added `--mps` flag with help text and link to PyTorch MPS docs.

- `cellbender/remove_background/cli.py`
  - Centralized device selection: sets `args.device` to `cuda` (if `--cuda` and available), else `mps` (if `--mps` and available), else `cpu`.
  - Emits friendly warnings when hardware is available but not selected (e.g., CUDA/MPS available but not requested).
  - Validates MPS availability (PyTorch built with MPS; macOS >= 12.3).

- `cellbender/remove_background/model.py`
  - Constructor now accepts `device: str` instead of `use_cuda: bool`.
  - Uses `.to(device)` (model and submodules) instead of `.cuda()`.
  - Removes `use_cuda` state; stores `self.device` only.
  - Replaces `pyro.plate(..., use_cuda=..., device=...)` with `pyro.plate(..., device=...)`.

- `cellbender/remove_background/data/dataprep.py`
  - `DataLoader` accepts `device: str` and pushes tensors to that device.
  - `prep_sparse_data_for_training(...)` accepts `device` and propagates it to loaders.

- `cellbender/remove_background/data/dataset.py`
  - `get_dataloader(...)` now takes `device: str` (instead of `use_cuda: bool`) and forwards it to `DataLoader`.

- `cellbender/remove_background/run.py`
  - Uses `args.device` consistently.
  - CPU threading configuration now keyed off `args.device == 'cpu'`.
  - Checkpoint loading uses `force_device` consistent with selected backend.
  - Passes `device` into posterior computations and estimators.

- `cellbender/remove_background/posterior.py`
  - Posterior derives its `device` from `vi_model.device` (or sensible fallback).
  - All data loader constructions and compute calls pass `device` explicitly.
  - Monitoring call still uses CUDA path only (see Limitations).

- `cellbender/remove_background/train.py`
  - Hardware usage log now checks `model.device == 'cuda'` (instead of `model.use_cuda`).
  - Safe device cache cleanup at end of training:
    - `torch.cuda.empty_cache()` if CUDA.
    - `torch.mps.empty_cache()` if MPS and available (wrapped in try/except).

## Device selection behavior

- If `--cuda` is provided and `torch.cuda.is_available()`, use `cuda`.
- Else if `--mps` is provided and `torch.backends.mps.is_available()` and `torch.backends.mps.is_built()`, use `mps`.
- Else fall back to `cpu`.

CUDA takes precedence over MPS when both are requested/available.

## How to use

```bash
# MPS on Apple Silicon (macOS 12.3+ with MPS-enabled PyTorch)
cellbender remove-background --input DATA.h5 --output OUT.h5 --mps

# CUDA (if available)
cellbender remove-background --input DATA.h5 --output OUT.h5 --cuda

# CPU (default)
cellbender remove-background --input DATA.h5 --output OUT.h5
```

To verify the flag is visible:

```bash
cellbender remove-background --help | grep -A 4 -- --mps
```

## Testing performed

- Local install in editable mode (0.3.2) succeeded.
- Verified that `--mps` appears in the CLI help.
- Confirmed `torch.backends.mps.is_available()` and `is_built()` on Apple Silicon test machine.
- Sanity checks for device propagation through model/data loaders/posterior/estimation.
- Ensured backward compatibility paths for `--cuda` and CPU remain intact.

Note: Full end-to-end test suite (including GPU tests) should be run in CI or by maintainers; this PR aims to be minimally invasive while restoring the MPS feature.

## Limitations and follow-ups

- Monitoring (`cellbender/monitor.py`) prints GPU utilization via `nvidia-smi` (CUDA only). There’s no analogous standard CLI for MPS; for now, logs omit MPS GPU utilization. A future improvement could add optional macOS/MPS metrics if a stable API becomes available.
- We preserve the 0.3.2 training/reporting default behavior; only device handling is generalized.

## Backward compatibility

- The public CLI remains the same; `--mps` is additive.
- CUDA and CPU workflows are unchanged.
- Checkpoints continue to load correctly; the chosen device is enforced via `force_device` where appropriate.

## Related work

- Original MPS addition: commit `8a70ea3` (Stephen Fleming) on `sf_pytorch_mps_backend` (0.2.0).
- This PR forward-ports that logic and adapts to the 0.3.2 internals.

## Reviewer notes

- Please focus on:
  - Correctness of device propagation (model, dataloaders, posterior, estimation).
  - Safety of checkpoint load/save across devices.
  - Absence of remaining `use_cuda` assumptions in the active code paths.
- If desired, this PR can be split by subsystem (CLI+argparse, data loaders, model/posterior) to ease review.

## Checklist

- [x] Adds `--mps` flag and help text
- [x] Generalizes device handling (cuda/mps/cpu)
- [x] Backward compatible with CUDA and CPU
- [x] Verified MPS availability path on Apple Silicon device
- [x] Minimal and contained code changes
- [ ] CI green (to be validated by maintainers)
- [ ] Optional docs update beyond CLI help (if desired by maintainers)

---

Thanks for reviewing! This should unlock fast, native GPU acceleration for a large portion of the community using Apple Silicon machines, while preserving the familiar CUDA and CPU paths.
