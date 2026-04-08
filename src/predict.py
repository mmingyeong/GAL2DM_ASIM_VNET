# models/unet/predict.py
# -*- coding: utf-8 -*-
"""
Predict with UNet3D (3D voxel-wise regression) and save to HDF5.

(A-option) Prediction uses the SAME DataLoader pipeline as training:
- Uses src.data_loader.get_dataloader(split="test") to resolve files, apply key-filtering,
  and apply the same normalization configuration as training.
- Augmentation is always OFF during prediction.

Supports channel-ablation inference:
  --input_case {both,ch1,ch2}
  --keep_two_channels  (keep in_channels=2 and zero-pad missing channel)

Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
Created: 2025-07-30
Last-Modified: 2025-12-18
"""

from __future__ import annotations

import os
import sys
import argparse
from contextlib import nullcontext
from typing import List, Sequence

import numpy as np
import torch
import h5py
from tqdm import tqdm

# project import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data_loader import get_dataloader
from src.model import UNet3D
from src.logger import get_logger

logger = get_logger("predict_unet3d")


# ----------------------------
# Helpers
# ----------------------------
def str2bool(v):
    return str(v).lower() in ("1", "true", "t", "yes", "y")


def select_inputs(x: torch.Tensor, case: str, keep_two: bool) -> torch.Tensor:
    """
    x: [B,2,D,H,W] (data_loader yields 2ch input by design)
    case: "both" | "ch1" | "ch2"
    keep_two=True  -> always return 2ch with zero-padding
    keep_two=False -> return 1ch for ch1/ch2 cases
    """
    assert x.ndim == 5 and x.size(1) == 2, f"Expected [B,2,D,H,W], got {tuple(x.shape)}"

    if case == "both":
        return x

    if case == "ch1":
        if keep_two:
            ch1 = x[:, 0:1]
            z = torch.zeros_like(ch1)
            return torch.cat([ch1, z], dim=1)
        return x[:, 0:1]

    if case == "ch2":
        if keep_two:
            ch2 = x[:, 1:2]
            z = torch.zeros_like(ch2)
            return torch.cat([z, ch2], dim=1)
        return x[:, 1:2]

    raise ValueError(f"Unknown input_case: {case}")


def _get_effective_file_paths_from_loader(loader) -> List[str]:
    """
    Robustly recover the file path list for the loader dataset, even if it is a Subset.

    Your data_loader returns:
      - dataset = ASIMHDF5Dataset(files, ...)
      - maybe wrapped by torch.utils.data.Subset(dataset, indices)

    We want the *exact order* of samples that the loader will iterate over
    (shuffle is False for test), so we can save prediction with matching filenames.

    Returns:
      list of absolute file paths, length == len(loader.dataset)
    """
    ds = loader.dataset

    # Case 1: Subset(ASIMHDF5Dataset, indices)
    if hasattr(ds, "dataset") and hasattr(ds, "indices"):
        base = ds.dataset
        indices = list(ds.indices)
        if hasattr(base, "file_paths"):
            base_paths = list(base.file_paths)
            return [base_paths[i] for i in indices]
        # Fallback: try attribute name variants
        for attr in ("files", "paths", "file_list"):
            if hasattr(base, attr):
                base_paths = list(getattr(base, attr))
                return [base_paths[i] for i in indices]
        raise AttributeError("Subset base dataset does not expose file paths (file_paths/files/paths).")

    # Case 2: plain ASIMHDF5Dataset
    if hasattr(ds, "file_paths"):
        return list(ds.file_paths)

    for attr in ("files", "paths", "file_list"):
        if hasattr(ds, attr):
            return list(getattr(ds, attr))

    raise AttributeError("Dataset does not expose file paths (file_paths/files/paths).")


def _load_checkpoint(model_path: str, device: torch.device):
    """Safe checkpoint load (supports plain state_dict or wrapped)."""
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)  # type: ignore
    except TypeError:
        state = torch.load(model_path, map_location=device)
    except Exception as e:
        logger.warning(f"weights_only load failed with {e}; falling back to standard torch.load")
        state = torch.load(model_path, map_location=device)

    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        elif "model" in state and isinstance(state["model"], dict):
            state = state["model"]
    return state


# ----------------------------
# Inference
# ----------------------------
def run_prediction(
    yaml_path: str,
    output_dir: str,
    model_path: str,
    device: str = "cuda",
    batch_size: int = 1,
    amp: bool = False,
    sample_fraction: float = 1.0,
    sample_seed: int = 42,
    input_case: str = "both",
    keep_two_channels: bool = False,
    validate_keys: bool = True,
    target_field: str = "rho",
    # include/exclude for prediction too (optional)
    exclude_list: str | None = None,
    include_list: str | None = None,
    # normalization / prediction-space flags
    normalize_input: bool = True,
    normalize_target: bool = False,
    eps: float = 1e-12,
):
    """
    Run inference on A-SIM test split using DataLoader.

    Notes:
      - Augmentation is forced OFF during prediction.
      - Normalization is configurable and should match training.
      - Even though prediction doesn't need y, data_loader returns (x, y). y is ignored.
    """
    if not (0 < sample_fraction <= 1.0):
        raise ValueError(f"--sample_fraction must be in (0,1], got {sample_fraction}")

    # case-specific subdir to avoid mixing outputs
    case_suffix = f"icase-{input_case}{'-keep2' if keep_two_channels else ''}"
    output_dir = os.path.join(output_dir, case_suffix)
    os.makedirs(output_dir, exist_ok=True)

    dev = torch.device(device)

    # Determine in_channels based on ablation config
    if input_case == "both":
        in_ch = 2
    else:
        in_ch = 2 if keep_two_channels else 1

    # Build model
    model = UNet3D(in_ch=in_ch, out_ch=1).to(dev)
    logger.info(f"🧱 Model: UNet3D(in_ch={in_ch}, out_ch=1) | input_case={input_case}, keep_two={keep_two_channels}")

    # Load checkpoint
    logger.info(f"📥 Loading checkpoint: {model_path}")
    state = _load_checkpoint(model_path, dev)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Missing keys while loading: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys while loading: {unexpected}")
    model.eval()

    # DataLoader (A-option)
    augmentation_cfg = {"enable": False}  # force OFF for prediction
    normalization_cfg = {
        "mode": "custom" if (normalize_input or normalize_target) else "none",
        "normalize_input": bool(normalize_input),
        "normalize_target": bool(normalize_target),
        "eps": float(eps),
    }

    test_loader = get_dataloader(
        yaml_path=yaml_path,
        split="test",
        batch_size=batch_size,
        shuffle=False,  # DO NOT shuffle in prediction
        sample_fraction=sample_fraction,
        num_workers=0,  # safer default for HDF5; override if you want
        pin_memory=True,
        target_field=target_field,          # only used for key validation + target loading
        dtype=torch.float32,
        seed=sample_seed,
        train_val_split=0.8,                # irrelevant for "test" split but required by API
        validate_keys=validate_keys,
        strict=False,
        exclude_list_path=exclude_list,
        include_list_path=include_list,
        augmentation=augmentation_cfg,
        normalization=normalization_cfg,
        apply_augmentation_in=(),           # ensure OFF
    )

    # Recover ordered file paths aligned with loader iteration order
    file_paths = _get_effective_file_paths_from_loader(test_loader)
    assert len(file_paths) == len(test_loader.dataset), "file path list length mismatch with dataset length"

    logger.info(f"🧪 Test samples: {len(test_loader.dataset)} (sample_fraction={sample_fraction})")
    logger.info(f"🧮 Normalization config (predict): {normalization_cfg}")

    # AMP context
    try:
        _ = torch.amp
        autocast_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float16) if (amp and dev.type == "cuda")
            else torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16) if (amp and dev.type == "cpu")
            else nullcontext()
        )
    except Exception:
        from torch.cuda.amp import autocast as legacy_autocast
        autocast_ctx = legacy_autocast(enabled=amp)

    # Predict
    saved_files: list[str] = []
    torch.set_grad_enabled(False)

    with torch.no_grad():
        for idx, (x, _y_unused) in enumerate(tqdm(test_loader, desc="🚀 Running UNet3D predictions")):
            src_path = file_paths[idx]
            filename = os.path.basename(src_path)
            output_path = os.path.join(output_dir, filename)
            if os.path.exists(output_path):
                logger.info(f"[SKIP] Already exists: {output_path}")
                continue
            x = x.to(dev, non_blocking=True)  # [B,2,D,H,W] from data_loader
            x = select_inputs(x, input_case, keep_two_channels)  # -> [B,in_ch,D,H,W]

            if x.size(1) != in_ch:
                raise RuntimeError(f"Post-selection channels {x.size(1)} != model.in_channels {in_ch} at idx={idx}")

            with autocast_ctx:
                pred = model(x)  # [B,1,D,H,W]

            y_pred = pred.float().cpu().numpy()
            y_pred = np.squeeze(y_pred, axis=1)  # (B,D,H,W)

            # Save
            with h5py.File(output_path, "w") as f_out:
                f_out.create_dataset("prediction", data=y_pred, compression="gzip")

                # Meta
                f_out.attrs["source_file"] = src_path
                f_out.attrs["model_path"] = model_path
                f_out.attrs["model_class"] = model.__class__.__name__
                f_out.attrs["amp"] = bool(amp)
                f_out.attrs["input_case"] = str(input_case)
                f_out.attrs["keep_two_channels"] = bool(keep_two_channels)
                f_out.attrs["normalization_mode"] = str(normalization_cfg["mode"])
                f_out.attrs["normalize_input"] = bool(normalization_cfg["normalize_input"])
                f_out.attrs["normalize_target"] = bool(normalization_cfg["normalize_target"])
                f_out.attrs["eps"] = float(normalization_cfg["eps"])

            saved_files.append(output_path)

    logger.info("====== UNet3D Inference Summary ======")
    logger.info(f"Saved files : {len(saved_files)}")
    if saved_files:
        logger.info("Saved (first 5): " + ", ".join(os.path.basename(p) for p in saved_files[:5]))


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UNet3D inference on A-SIM test split (DataLoader-based).")

    # Data / Paths
    parser.add_argument("--yaml_path", type=str, required=True, help="Path to asim_paths.yaml")
    parser.add_argument("--output_dir", type=str, required=True, help="Root directory to save predictions (per-case subdir will be created)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .pt file")

    # System
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--amp", action="store_true", help="Enable mixed-precision inference")

    # Subsampling (file-level, since dataset is file-based)
    parser.add_argument("--sample_fraction", type=float, default=1.0,
                        help="Fraction (0,1] of TEST FILES to run (dataset is file-based).")
    parser.add_argument("--sample_seed", type=int, default=42,
                        help="Random seed for reproducible file-level subsampling.")

    # Channel ablation flags
    parser.add_argument("--input_case", type=str, choices=["both", "ch1", "ch2"], default="both",
                        help="Select which input channels are provided to the model.")
    parser.add_argument("--keep_two_channels", action="store_true",
                        help="If set, keep in_channels=2 and zero-pad the missing channel for single-channel cases.")

    # DataLoader validation & lists
    parser.add_argument("--validate_keys", type=str2bool, default=True,
                        help="Pre-scan HDF5 to check required keys (input/output_*). Set False to skip (faster).")
    parser.add_argument("--exclude_list", type=str, default=None,
                        help="Path to text file containing bad HDF5 file paths to exclude.")
    parser.add_argument("--include_list", type=str, default=None,
                        help="Path to text file containing good HDF5 file paths to include only.")

    # Normalization (should match training)
    parser.add_argument("--target_field", type=str, choices=["rho", "tscphi"], default="rho",
                        help="Used only for dataloader key-validation/target read; prediction itself uses only x.")
    parser.add_argument("--normalize_input", type=str2bool, default=True,
                        help="Apply custom input normalization (vpec -> [-1,1]). Should match training.")
    parser.add_argument("--normalize_target", type=str2bool, default=False,
                        help="Apply custom target normalization. For UNet-style regression, set this to match training.")
    parser.add_argument("--eps", type=float, default=1e-12)

    args = parser.parse_args()

    run_prediction(
        yaml_path=args.yaml_path,
        output_dir=args.output_dir,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        amp=args.amp,
        sample_fraction=args.sample_fraction,
        sample_seed=args.sample_seed,
        input_case=args.input_case,
        keep_two_channels=args.keep_two_channels,
        validate_keys=args.validate_keys,
        target_field=args.target_field,
        exclude_list=args.exclude_list,
        include_list=args.include_list,
        normalize_input=args.normalize_input,
        normalize_target=args.normalize_target,
        eps=args.eps,
    )
