# models/unet/predict.py
# -*- coding: utf-8 -*-
"""
Predict with UNet3D (3D voxel-wise regression) and save to HDF5.

Prediction uses the SAME DataLoader pipeline as training:
- Uses src.data_loader.get_dataloader(split="test") to resolve files, apply key-filtering,
  and apply the same normalization configuration as training.
- Augmentation is always OFF during prediction.

Supports channel-ablation inference:
  --input_case {both,ch1,ch2}
  --keep_two_channels  (keep in_channels=2 and zero-pad missing channel)
"""

from __future__ import annotations

import os
import sys
import argparse
from contextlib import nullcontext
from typing import List

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
DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def str2bool(v):
    return str(v).lower() in ("1", "true", "t", "yes", "y")


def select_inputs(x: torch.Tensor, case: str, keep_two: bool) -> torch.Tensor:
    """
    x: [B,2,D,H,W]
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
    Recover file path list in the exact order of loader iteration.
    """
    ds = loader.dataset

    if hasattr(ds, "dataset") and hasattr(ds, "indices"):
        base = ds.dataset
        indices = list(ds.indices)

        if hasattr(base, "file_paths"):
            base_paths = list(base.file_paths)
            return [base_paths[i] for i in indices]

        for attr in ("files", "paths", "file_list"):
            if hasattr(base, attr):
                base_paths = list(getattr(base, attr))
                return [base_paths[i] for i in indices]

        raise AttributeError("Subset base dataset does not expose file paths.")

    if hasattr(ds, "file_paths"):
        return list(ds.file_paths)

    for attr in ("files", "paths", "file_list"):
        if hasattr(ds, attr):
            return list(getattr(ds, attr))

    raise AttributeError("Dataset does not expose file paths.")


def _load_checkpoint(model_path: str, device: torch.device):
    """
    Safe checkpoint load.
    Supports plain state_dict or wrapped checkpoint.
    """
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
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
    base_channels: int = 32,
    batch_size: int = 1,
    amp: bool = False,
    sample_fraction: float = 1.0,
    sample_seed: int = 42,
    input_case: str = "both",
    keep_two_channels: bool = False,
    validate_keys: bool = True,
    target_field: str = "rho",
    exclude_list: str | None = None,
    include_list: str | None = None,
    normalize_input: bool = True,
    normalize_target: bool = False,
    eps: float = 1e-12,
    dtype: str = "float32",
    save_dtype: str = "float32",
):
    """
    Run inference on A-SIM test split using DataLoader.

    Notes:
      - Augmentation is forced OFF during prediction.
      - Normalization is configurable and should match training.
      - base_channels must match the trained VNet/UNet checkpoint width.
    """
    if not (0 < sample_fraction <= 1.0):
        raise ValueError(f"--sample_fraction must be in (0,1], got {sample_fraction}")

    if dtype not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype}. Choose from {list(DTYPE_MAP.keys())}")

    if save_dtype not in ("float32", "float64"):
        raise ValueError("save_dtype must be one of ['float32', 'float64']")

    torch_dtype = DTYPE_MAP[dtype]
    np_save_dtype = np.float32 if save_dtype == "float32" else np.float64

    case_suffix = f"icase-{input_case}{'-keep2' if keep_two_channels else ''}"
    output_dir = os.path.join(output_dir, case_suffix)
    os.makedirs(output_dir, exist_ok=True)

    dev = torch.device(device)

    if input_case == "both":
        in_ch = 2
    else:
        in_ch = 2 if keep_two_channels else 1

    # --------------------------------------------------
    # Build model
    # IMPORTANT: base_channels must match checkpoint.
    # --------------------------------------------------
    model = UNet3D(
        in_ch=in_ch,
        out_ch=1,
        BASE=base_channels,
    ).to(device=dev, dtype=torch_dtype)

    logger.info(
        f"🧱 Model: UNet3D(in_ch={in_ch}, out_ch=1, base={base_channels}) | "
        f"input_case={input_case}, keep_two={keep_two_channels} | dtype={torch_dtype}"
    )

    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------
    logger.info(f"📥 Loading checkpoint: {model_path}")
    state = _load_checkpoint(model_path, dev)

    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        logger.warning(f"Missing keys while loading: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys while loading: {unexpected}")

    model.eval()

    # --------------------------------------------------
    # DataLoader
    # --------------------------------------------------
    augmentation_cfg = {"enable": False}

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
        shuffle=False,
        sample_fraction=sample_fraction,
        num_workers=0,
        pin_memory=True,
        target_field=target_field,
        dtype=torch_dtype,
        seed=sample_seed,
        train_val_split=0.8,
        validate_keys=validate_keys,
        strict=False,
        exclude_list_path=exclude_list,
        include_list_path=include_list,
        augmentation=augmentation_cfg,
        normalization=normalization_cfg,
        apply_augmentation_in=(),
    )

    file_paths = _get_effective_file_paths_from_loader(test_loader)

    assert len(file_paths) == len(test_loader.dataset), (
        "file path list length mismatch with dataset length"
    )

    logger.info(f"🧪 Test samples: {len(test_loader.dataset)} (sample_fraction={sample_fraction})")
    logger.info(f"🧮 Normalization config (predict): {normalization_cfg}")

    # --------------------------------------------------
    # AMP
    # --------------------------------------------------
    use_amp = bool(amp and dev.type == "cuda" and torch_dtype == torch.float32)

    if amp and torch_dtype != torch.float32:
        logger.warning("⚠️ AMP requested, but dtype is not float32. AMP will be disabled.")

    try:
        if use_amp:
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        else:
            autocast_ctx = nullcontext()
    except Exception:
        from torch.cuda.amp import autocast as legacy_autocast
        autocast_ctx = legacy_autocast(enabled=use_amp)

    # --------------------------------------------------
    # Predict
    # --------------------------------------------------
    saved_files: list[str] = []
    skipped_files: list[str] = []

    torch.set_grad_enabled(False)

    sample_offset = 0

    with torch.no_grad():
        for batch_idx, (x, _y_unused) in enumerate(
            tqdm(test_loader, desc="🚀 Running UNet3D predictions")
        ):
            batch_size_actual = x.size(0)

            x = x.to(device=dev, dtype=torch_dtype, non_blocking=True)
            x = select_inputs(x, input_case, keep_two_channels)

            if batch_idx == 0:
                logger.info(
                    f"🔎 dtype check | model={next(model.parameters()).dtype} | "
                    f"input={x.dtype} | save_dtype={save_dtype} | "
                    f"batch_size={batch_size_actual}"
                )

            if x.size(1) != in_ch:
                raise RuntimeError(
                    f"Post-selection channels {x.size(1)} != model.in_channels {in_ch} "
                    f"at batch_idx={batch_idx}"
                )

            with autocast_ctx:
                pred = model(x)

            y_pred = pred.detach().cpu().numpy().astype(np_save_dtype, copy=False)
            y_pred = np.squeeze(y_pred, axis=1)

            if y_pred.shape[0] != batch_size_actual:
                raise RuntimeError(
                    f"Prediction batch mismatch: y_pred.shape[0]={y_pred.shape[0]} "
                    f"vs batch_size_actual={batch_size_actual}"
                )

            batch_file_paths = file_paths[sample_offset : sample_offset + batch_size_actual]

            if len(batch_file_paths) != batch_size_actual:
                raise RuntimeError(
                    f"file_paths slice mismatch at sample_offset={sample_offset}: "
                    f"expected {batch_size_actual}, got {len(batch_file_paths)}"
                )

            for b, src_path in enumerate(batch_file_paths):
                filename = os.path.basename(src_path)
                output_path = os.path.join(output_dir, filename)

                if os.path.exists(output_path):
                    logger.info(f"[SKIP] Already exists: {output_path}")
                    skipped_files.append(output_path)
                    continue

                pred_i = y_pred[b]

                with h5py.File(output_path, "w") as f_out:
                    f_out.create_dataset("prediction", data=pred_i, compression="gzip")

                    f_out.attrs["source_file"] = src_path
                    f_out.attrs["model_path"] = model_path
                    f_out.attrs["model_class"] = model.__class__.__name__
                    f_out.attrs["base_channels"] = int(base_channels)
                    f_out.attrs["amp"] = bool(use_amp)
                    f_out.attrs["input_case"] = str(input_case)
                    f_out.attrs["keep_two_channels"] = bool(keep_two_channels)
                    f_out.attrs["normalization_mode"] = str(normalization_cfg["mode"])
                    f_out.attrs["normalize_input"] = bool(normalization_cfg["normalize_input"])
                    f_out.attrs["normalize_target"] = bool(normalization_cfg["normalize_target"])
                    f_out.attrs["eps"] = float(normalization_cfg["eps"])
                    f_out.attrs["dtype"] = str(dtype)
                    f_out.attrs["save_dtype"] = str(save_dtype)
                    f_out.attrs["batch_index"] = int(batch_idx)
                    f_out.attrs["batch_pos"] = int(b)
                    f_out.attrs["global_sample_index"] = int(sample_offset + b)

                saved_files.append(output_path)

            sample_offset += batch_size_actual

    logger.info("====== UNet3D Inference Summary ======")
    logger.info(f"Saved files   : {len(saved_files)}")
    logger.info(f"Skipped files : {len(skipped_files)}")
    logger.info(f"Total handled : {sample_offset}")

    if saved_files:
        logger.info("Saved first 5: " + ", ".join(os.path.basename(p) for p in saved_files[:5]))


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run UNet3D inference on A-SIM test split."
    )

    # Data / Paths
    parser.add_argument("--yaml_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)

    # System
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--amp", action="store_true")

    # Model width
    parser.add_argument(
        "--base_channels",
        type=int,
        default=32,
        help="Base channel width of UNet3D. Must match the training checkpoint.",
    )

    # dtype
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float32",
    )
    parser.add_argument(
        "--save_dtype",
        type=str,
        choices=["float32", "float64"],
        default="float32",
    )

    # Subsampling
    parser.add_argument("--sample_fraction", type=float, default=1.0)
    parser.add_argument("--sample_seed", type=int, default=42)

    # Channel ablation
    parser.add_argument(
        "--input_case",
        type=str,
        choices=["both", "ch1", "ch2"],
        default="both",
    )
    parser.add_argument("--keep_two_channels", action="store_true")

    # DataLoader validation & lists
    parser.add_argument("--validate_keys", type=str2bool, default=True)
    parser.add_argument("--exclude_list", type=str, default=None)
    parser.add_argument("--include_list", type=str, default=None)

    # Normalization
    parser.add_argument(
        "--target_field",
        type=str,
        choices=["rho", "tscphi"],
        default="rho",
    )
    parser.add_argument("--normalize_input", type=str2bool, default=True)
    parser.add_argument("--normalize_target", type=str2bool, default=False)
    parser.add_argument("--eps", type=float, default=1e-12)

    args = parser.parse_args()

    run_prediction(
        yaml_path=args.yaml_path,
        output_dir=args.output_dir,
        model_path=args.model_path,
        device=args.device,
        base_channels=args.base_channels,
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
        dtype=args.dtype,
        save_dtype=args.save_dtype,
    )