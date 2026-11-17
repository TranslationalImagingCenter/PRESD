"""Run a trained model on one input .lsm file and save the 32-channel prediction.

The input file (and optional GT file) are made by prepare_data.py:
    input: (R, C_in, H, W)    GT: (32, H, W)
C_in and R are read from the input file and must match the model
(e.g. NN_4ch_5repeats.pt for C_in = 4, R = 5). If a GT file is given,
metrics, error maps and GT comparisons are saved too.
"""
import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from tqdm import tqdm

from config import load_global_params, set_channel_num
from NN import PRESD, phasor_loss
from helper import normalize_data, invert_scale, compute_mape, compute_smape, set_seeds, file_to_block_number, get_device

global_params = load_global_params()

NUM_SPECTRA_PLOTS = 50
SPECTRA_SEED = 31


def read_input(file_path: str) -> np.ndarray:
    """Read a prepared input file as (R, C_in, H, W)."""
    data = tifffile.TiffFile(file_path).asarray()
    if data.ndim != 4:
        raise ValueError(f"Expected input shape (R, C_in, H, W) in {file_path}, got {data.shape}. "
                         "Create input files with prepare_data.py.")
    return data


def _prepare_file_tensors(
    data: np.ndarray,
    norm_type: str,
    wavelength: float,
    gt_file_path: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Tuple[int, int]]:
    """Build normalized per-pixel arrays from an input array (and optional GT .lsm).

    Args:
        data: input array, (R, C_in, H, W)
    Returns:
        x_data: (N, 1, C_in, R)   N = H * W
        y_true: (N, 32), or None if no GT file is given
        w_data: (N,) excitation wavelength in nm
        spatial_shape: (H, W)
    """
    R, C, H, W = data.shape
    M = H * W
    x_in = data.reshape(R, C, M).transpose(1, 0, 2).astype(np.float32)   # (C_in, R, M)

    if gt_file_path is not None:
        gt_data = tifffile.TiffFile(gt_file_path).asarray()
        if gt_data.shape != (global_params["OUTPUT_CHANNELS"], H, W):
            raise ValueError(f"GT shape {gt_data.shape} does not match input {data.shape}; "
                             f"expected ({global_params['OUTPUT_CHANNELS']}, {H}, {W})")
        y_out = gt_data.reshape(global_params["OUTPUT_CHANNELS"], M).astype(np.float32)  # (32, M)
        x_in, y_out = normalize_data(x_in, y_out, norm_type)
        y_true = y_out.T.astype(np.float32)
    else:
        x_in = normalize_data(x_in, norm_type=norm_type)
        y_true = None

    x_data = np.transpose(x_in, (2, 0, 1))[:, np.newaxis].astype(np.float32)  # (N, 1, C, R)
    w_data = np.full((M,), float(wavelength), dtype=np.float32)
    return x_data, y_true, w_data, (H, W)


def _batched_infer(model, device, x, w, batch_size) -> np.ndarray:
    """Run the model over all pixels in batches; returns (N, 32)."""
    model.eval()
    preds: List[np.ndarray] = []
    with torch.no_grad():
        with tqdm(total=x.shape[0], unit="px", desc="Inference", ncols=80) as pbar:
            for start in range(0, x.shape[0], batch_size):
                end = min(start + batch_size, x.shape[0])
                x_t = torch.tensor(x[start:end], dtype=torch.float32, device=device)
                w_t = torch.tensor(w[start:end], dtype=torch.float32, device=device)
                preds.append(model(x_t, w_t).cpu().numpy())
                pbar.update(end - start)
    return np.vstack(preds)


def _compute_phasor_map(gt_32: np.ndarray, pred_32: np.ndarray, harmonic: int) -> np.ndarray:
    """Per-pixel phasor (G, S) distance between GT and prediction. Inputs: (32, H, W)."""
    C = gt_32.shape[0]
    angles = 2.0 * np.pi * harmonic * np.arange(C, dtype=np.float32) / float(C)
    cos_a = np.cos(angles).reshape(C, 1, 1)
    sin_a = np.sin(angles).reshape(C, 1, 1)
    eps = global_params["PHASOR_EPSILON"]

    sum_gt = gt_32.sum(axis=0) + eps
    sum_pr = pred_32.sum(axis=0) + eps
    G_gt = (gt_32 * cos_a).sum(axis=0) / sum_gt
    S_gt = (gt_32 * sin_a).sum(axis=0) / sum_gt
    G_pr = (pred_32 * cos_a).sum(axis=0) / sum_pr
    S_pr = (pred_32 * sin_a).sum(axis=0) / sum_pr
    return np.sqrt((G_pr - G_gt) ** 2 + (S_pr - S_gt) ** 2)


def _save_spectra_plots(out_dir, x, pred_img_32, gt_img_32, ph_map, norm_type):
    """Plot input, prediction and (optional) GT spectra for random pixels."""
    os.makedirs(out_dir, exist_ok=True)
    n_out, H, W = pred_img_32.shape
    channel_num = global_params["channel_num"]
    bin_edges = np.arange(channel_num + 1) * (n_out // channel_num)
    eps = global_params["MAPE_EPSILON"]

    rng = np.random.default_rng(SPECTRA_SEED)
    coords = list(zip(rng.integers(0, H, NUM_SPECTRA_PLOTS), rng.integers(0, W, NUM_SPECTRA_PLOTS)))

    for i, (hh, ww) in enumerate(coords, start=1):
        pr_spec = pred_img_32[:, hh, ww]
        # Input intensities averaged over repeats, drawn as one bar per input channel.
        inp_vals = invert_scale(x[hh * W + ww, 0], norm_type=norm_type).mean(axis=-1)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.stairs(inp_vals, bin_edges, fill=True, alpha=0.2, color='green', label='Input')
        ax.stairs(inp_vals, bin_edges, color='green', linewidth=1.5)
        ax.plot(range(n_out), pr_spec, 'r--', label='Pred', linewidth=2)

        if gt_img_32 is not None:
            gt_spec = gt_img_32[:, hh, ww]
            spec_rmse = float(np.sqrt(np.mean((pr_spec - gt_spec) ** 2)))
            spec_smape = float(np.mean(2 * np.abs(pr_spec - gt_spec) / (np.abs(gt_spec) + np.abs(pr_spec) + eps)) * 100)
            ax.plot(range(n_out), gt_spec, 'b-', label='GT', linewidth=2)
            ax.set_title(f'Spectrum ({hh},{ww})\nRMSE={spec_rmse:.1f}  SMAPE={spec_smape:.1f}%  '
                         f'Phasor h2={float(ph_map[hh, ww]):.4f}')
        else:
            ax.set_title(f'Spectrum ({hh},{ww})')

        ax.set_xlabel('Output Channel')
        ax.set_ylabel('Intensity')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"spectrum_{i:02d}_{hh}_{ww}.png"), dpi=130)
        plt.close(fig)


def _save_channel_images(out_dir, pred_img_32, gt_img_32, ph_map):
    """Save one figure per output channel (with GT, SMAPE, abs. error and phasor maps if GT is given)."""
    os.makedirs(out_dir, exist_ok=True)
    eps = global_params["MAPE_EPSILON"]

    for ch in range(pred_img_32.shape[0]):
        pr_ch = np.nan_to_num(pred_img_32[ch], nan=0.0, posinf=0.0, neginf=0.0)

        if gt_img_32 is not None:
            gt_ch = np.nan_to_num(gt_img_32[ch], nan=0.0, posinf=0.0, neginf=0.0)
            smape_ch = np.clip(2 * np.abs(pr_ch - gt_ch) / (np.abs(gt_ch) + np.abs(pr_ch) + eps) * 100.0, 0, 50)
            abs_err_ch = np.abs(pr_ch - gt_ch)
            panels = [gt_ch, pr_ch, smape_ch, abs_err_ch, ph_map]
            titles = [f'GT ch {ch+1}', f'Pred ch {ch+1}', 'SMAPE (%)', 'Abs. error', 'Phasor h2']
            cmaps = ['gray', 'gray', 'magma', 'magma', 'viridis']
            vmins = [None, None, 0, 0, 0]
            vmaxs = [None, None, 50, np.max(pr_ch) / 5.0, 0.15]
        else:
            panels, titles, cmaps, vmins, vmaxs = [pr_ch], [f'Pred ch {ch+1}'], ['gray'], [None], [None]

        n_cols = len(panels)
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        axes = np.atleast_1d(axes)
        for ax, panel, title, cmap, vmin, vmax in zip(axes, panels, titles, cmaps, vmins, vmaxs):
            im = ax.imshow(panel, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"channel_{ch+1:02d}.png"), dpi=140, bbox_inches='tight')
        plt.close(fig)


def evaluate_on_file(
    model: torch.nn.Module,
    device: torch.device,
    file_path: str,
    output_dir: str,
    wavelength: float,
    gt_file: Optional[str] = None,
) -> Dict[str, float]:
    """Predict all pixels of one .lsm file and write the outputs to output_dir.

    Writes pred32.lsm, spectra/ and channels/; with a GT file also metrics.json.
    Returns the metrics dict (empty if no GT file is given).
    """
    if not os.path.isfile(file_path):
        raise ValueError(f"Input file not found: {file_path}")
    if not file_path.endswith(global_params["FILE_EXTENSION"]):
        raise ValueError(f"Expected a {global_params['FILE_EXTENSION']} file, got: {file_path}")
    if gt_file is not None and not os.path.isfile(gt_file):
        raise ValueError(f"GT file not found: {gt_file}")

    os.makedirs(output_dir, exist_ok=True)
    norm_type = global_params["NORM"]
    ph_harm = global_params["PHASOR_HARMONICS"]
    n_out = global_params["OUTPUT_CHANNELS"]

    print(f"\nProcessing {file_path}")
    print("  [1/6] Loading and preparing tensors...")
    t0 = time.time()
    x, y_true, w, (H, W) = _prepare_file_tensors(
        read_input(file_path), norm_type, wavelength=wavelength, gt_file_path=gt_file,
    )
    print(f"        Done in {time.time() - t0:.1f}s  |  pixels: {x.shape[0]}  shape: {(H, W)}")

    print("  [2/6] Running inference...")
    t0 = time.time()
    y_pred = _batched_infer(model, device, x, w, global_params["INFERENCE_BATCH_SIZE"])
    elapsed = time.time() - t0
    print(f"        Done in {elapsed:.1f}s  ({x.shape[0] / elapsed:.0f} pixels/s)")

    print("  [3/6] Computing metrics...")
    t0 = time.time()
    metrics: Dict[str, float] = {}
    if y_true is not None:
        y_pred_orig, y_true_orig = invert_scale(y_pred, y_true, norm_type)
        mean_smape, median_smape = compute_smape(y_true_orig, y_pred_orig)
        yt = torch.tensor(y_true_orig, dtype=torch.float32)
        yp = torch.tensor(y_pred_orig, dtype=torch.float32)
        metrics = {
            "mape": float(compute_mape(y_true_orig, y_pred_orig)),
            "mean_smape": mean_smape,
            "median_smape": median_smape,
            "rmse": float(np.sqrt(np.mean((y_true_orig - y_pred_orig) ** 2))),
            "phasor_h1": float(phasor_loss(yp, yt, harmonic=ph_harm[0])),
            "phasor_h2": float(phasor_loss(yp, yt, harmonic=ph_harm[1])),
        }
        print(f"        Done in {time.time() - t0:.1f}s  |  MAPE={metrics['mape']:.2f}%  "
              f"SMAPE={mean_smape:.2f}%  RMSE={metrics['rmse']:.1f}  "
              f"ph1={metrics['phasor_h1']:.4f}  ph2={metrics['phasor_h2']:.4f}")
    else:
        y_pred_orig = invert_scale(y_pred, norm_type=norm_type)
        y_true_orig = None
        print("        No GT file — skipping metrics")

    print("  [4/6] Saving metrics.json...")
    if metrics:
        with open(os.path.join(output_dir, "metrics.json"), "w") as jf:
            json.dump(metrics, jf, indent=global_params["JSON_INDENT"])
    else:
        print("        No GT — skipping metrics.json")

    print("  [5/6] Saving prediction LSM and spectra plots...")
    t0 = time.time()
    pred_img_32 = y_pred_orig.T.reshape(n_out, H, W)
    gt_img_32 = y_true_orig.T.reshape(n_out, H, W) if y_true_orig is not None else None
    tifffile.imwrite(os.path.join(output_dir, "pred32.lsm"), pred_img_32)

    try:
        ph_map = _compute_phasor_map(gt_img_32, pred_img_32, harmonic=ph_harm[1]) if gt_img_32 is not None else None
        _save_spectra_plots(os.path.join(output_dir, "spectra"), x, pred_img_32, gt_img_32, ph_map, norm_type)
        print(f"        Done in {time.time() - t0:.1f}s")

        print(f"  [6/6] Saving channel images ({n_out} channels)...")
        t0 = time.time()
        _save_channel_images(os.path.join(output_dir, "channels"), pred_img_32, gt_img_32, ph_map)
        print(f"        Done in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"Warning: saving plots failed for {file_path}: {e}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run a trained model on one input .lsm file made by prepare_data.py")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Model .pt file: one of models/NN_<C>ch_<R>repeats.pt or your own from train.py")
    parser.add_argument("--input_file", type=str, required=True, help="Input .lsm file, shape (R, C_in, H, W)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Folder where all outputs are written "
                             f"(default: {global_params['PLOTS_DIR']}/test/<input file name>)")
    parser.add_argument("--wavelength", type=float, default=None,
                        help="Excitation wavelength in nm (e.g. 405, 458, 488, 514, 561, 591, 633). "
                             "Default: read from 'Block N' in the input filename (WAVELENGTH_BLOCK_N in global_val.txt)")
    parser.add_argument("--gt_file", type=str, default=None,
                        help="Optional 32-channel GT .lsm file. If omitted, only the prediction is saved (no metrics)")
    parser.add_argument("--seed", type=int, default=global_params["SEED"], help="Random seed")
    args = parser.parse_args()

    # C_in and R come from the input file; the model must have been trained for the same values.
    shape = tifffile.TiffFile(args.input_file).series[0].shape
    if len(shape) != 4:
        raise ValueError(f"Expected input shape (R, C_in, H, W), got {shape}. Create input files with prepare_data.py.")
    num_repeats, channel_num = shape[0], shape[1]
    set_channel_num(channel_num)
    print(f"Input: C_in={channel_num}, R={num_repeats}, H×W={shape[2]}×{shape[3]}")

    if args.output_dir is None:
        input_name = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output_dir = os.path.join(global_params["PLOTS_DIR"], "test", input_name)
    print(f"Outputs will be saved to: {os.path.abspath(args.output_dir)}")

    # Excitation wavelength: from "Block N" in the filename, unless given explicitly.
    block = file_to_block_number(os.path.basename(args.input_file))
    file_wavelength = global_params["WAVELENGTH_MAP"].get(block)
    if args.wavelength is None:
        if file_wavelength is None:
            raise SystemExit(f"Could not read a block number (1-7) from {args.input_file}; pass --wavelength.")
        args.wavelength = float(file_wavelength)
        print(f"Wavelength: {args.wavelength:.0f} nm (Block {block} in filename)")
    else:
        print(f"Wavelength: {args.wavelength:.0f} nm (from --wavelength)")
        if file_wavelength is not None and float(file_wavelength) != args.wavelength:
            print(f"Warning: filename says Block {block} = {file_wavelength} nm, "
                  f"but --wavelength is {args.wavelength:.0f} nm")

    set_seeds(args.seed)
    device = get_device()
    print(f"Device: {device}")

    model = PRESD(replicate_count=num_repeats)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    try:
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    except RuntimeError as e:
        raise SystemExit(f"\nModel {args.model_path} does not fit this input (C_in={channel_num}, R={num_repeats}). "
                         f"Use a model trained for {channel_num} channels and {num_repeats} repeats, "
                         f"e.g. models/NN_{channel_num}ch_{num_repeats}repeats.pt.\n\nDetails: {e}")
    model.to(device)

    metrics = evaluate_on_file(
        model=model,
        device=device,
        file_path=args.input_file,
        output_dir=args.output_dir,
        wavelength=args.wavelength,
        gt_file=args.gt_file,
    )

    if metrics:
        print(f"Metrics -> MAPE: {metrics['mape']:.4f}% | Mean_SMAPE: {metrics['mean_smape']:.4f}% | "
              f"Median_SMAPE: {metrics['median_smape']:.4f}% | RMSE: {metrics['rmse']:.6f} | "
              f"Phasor h1: {metrics['phasor_h1']:.6f} | Phasor h2: {metrics['phasor_h2']:.6f}")
    else:
        print("Inference complete. No GT provided — metrics not computed.")
    print(f"Outputs saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
