import os
import argparse
import json
import time
from typing import List, Tuple, Dict

import numpy as np
import torch
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import load_global_params
from NN import FilmEarlyNet, phasor_loss
from helper import (
    normalize_data,
    invert_scale,
    file_to_wavelength_norm,
    compute_mape,
    compute_smape,
    initialize_directories,
    set_seeds,
)


global_params = load_global_params()

def _prepare_file_tensors(
    file_path: str,
    repeats: List[int],
    channel_num: int,
    norm_type: str,
    wavelength: float,
    gt_file_path: str = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """Load a single .lsm input file (natively channel_num channels) and produce
    normalized tensors for inference over every pixel.  Ground truth is loaded
    from gt_file_path if provided (a separate 32-channel .lsm file); otherwise
    y_true is None.

    Returns:
        x_data: (N, 1, channel_num, R)
        y_true: (N, 32)  or  None if no GT file given
        w_data: (N,)
        spatial_shape: (H, W)
    """
    # --- Input file: natively channel_num channels ---
    data = tifffile.TiffFile(file_path).asarray()
    # Actual file shape: (num_repeats, channel_num, H, W)
    H, W = data.shape[-2], data.shape[-1]
    M = H * W
    num_repeats_in_file = data.shape[0]
    wave_norm = float(wavelength)

    # Build input: data[i] gives (channel_num, H, W) for repeat i
    pos_list = repeats
    raw_in: List[np.ndarray] = []
    for i in pos_list:
        if i >= num_repeats_in_file:
            break
        raw_in.append(data[i].reshape(channel_num, -1))  # (channel_num, M)
    if len(raw_in) < len(pos_list):
        raise ValueError(f"Not enough repeats in {file_path} (has {num_repeats_in_file}) for repeats {pos_list}")

    compressed_in = np.zeros((channel_num, len(pos_list), M), dtype=np.float32)
    for p_idx, slice_ch in enumerate(raw_in):
        compressed_in[:, p_idx] = slice_ch

    # --- GT file: shape is (32, H, W) ---
    if gt_file_path is not None:
        gt_data = tifffile.TiffFile(gt_file_path).asarray()
        out_32 = gt_data.reshape(global_params["OUTPUT_CHANNELS"], -1).astype(np.float32)  # (32, M)
    else:
        out_32 = None

    file_in = compressed_in
    file_out = out_32  # may be None (inference-only mode)
    file_wave = np.full((M,), wave_norm, dtype=np.float32)

    # Normalize — if no GT, normalize input only
    if file_out is not None:
        file_in, file_out = normalize_data(file_in, file_out, norm_type)
    else:
        file_in, _ = normalize_data(file_in, np.zeros((global_params["OUTPUT_CHANNELS"], file_in.shape[2]), dtype=np.float32), norm_type)

    # Reshape for model: (N, 1, C, R)
    N = file_in.shape[2]
    x_data = np.transpose(file_in, (2, 0, 1))   # (N, C, R)
    x_data = np.expand_dims(x_data, axis=1)      # (N, 1, C, R)
    y_true = file_out.T.astype(np.float32) if file_out is not None else None

    return x_data.astype(np.float32), y_true, file_wave.astype(np.float32), (H, W)


def _batched_infer(
    model: torch.nn.Module,
    device: torch.device,
    x: np.ndarray,
    w: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Run inference in batches; returns predictions as numpy (N, 32)."""
    model.eval()
    preds: List[np.ndarray] = []
    with torch.no_grad():
        with tqdm(total=x.shape[0], unit="px", desc="Inference", ncols=80) as pbar:
            for start in range(0, x.shape[0], batch_size):
                end = min(start + batch_size, x.shape[0])
                x_t = torch.tensor(x[start:end], dtype=torch.float32, device=device)
                w_t = torch.tensor(w[start:end], dtype=torch.float32, device=device)
                y_hat = model(x_t, w_t).detach().cpu().numpy()
                preds.append(y_hat)
                pbar.update(end - start)
    return np.vstack(preds)


# -------------------------
# Per-pixel map helpers
# -------------------------
def _compute_mape_map(gt_32: np.ndarray, pred_32: np.ndarray) -> np.ndarray:
    """Compute per-pixel MAPE (%) over spectral dimension.
    gt_32, pred_32: (32, H, W)"""
    eps = 1e-10
    diff = np.abs(pred_32 - gt_32)
    denom = np.abs(gt_32) + eps
    mape_map = np.mean(diff / denom, axis=0) * 100.0
    return mape_map


def _compute_phasor_h2_map(gt_32: np.ndarray, pred_32: np.ndarray, harmonic: int = 2) -> np.ndarray:
    """Compute phasor (G,S) distance per pixel at given harmonic.
    Inputs are (32, H, W)."""
    C, H, W = gt_32.shape
    idx = np.arange(C, dtype=np.float32)
    angles = 2.0 * np.pi * harmonic * idx / float(C)
    cos_a = np.cos(angles).reshape(C, 1, 1)
    sin_a = np.sin(angles).reshape(C, 1, 1)

    eps = 1e-12
    sum_gt = gt_32.sum(axis=0, keepdims=True) + eps
    sum_pr = pred_32.sum(axis=0, keepdims=True) + eps

    G_gt = (gt_32 * cos_a).sum(axis=0) / sum_gt.squeeze(0)
    S_gt = (gt_32 * sin_a).sum(axis=0) / sum_gt.squeeze(0)
    G_pr = (pred_32 * cos_a).sum(axis=0) / sum_pr.squeeze(0)
    S_pr = (pred_32 * sin_a).sum(axis=0) / sum_pr.squeeze(0)

    return np.sqrt((G_pr - G_gt) ** 2 + (S_pr - S_gt) ** 2)


def evaluate_on_file(
    model: torch.nn.Module,
    device: torch.device,
    repeats: List[int],
    file_path: str,
    output_dir: str,
    wavelength: float,
    gt_file: str = None,
) -> Dict[str, float]:
    """Evaluate model on every pixel of a single .lsm file. Saves outputs to output_dir.
    If gt_file is provided, metrics (MAPE, RMSE, phasor) are also computed and saved.

    Returns metrics dict (empty if no GT file given).
    """
    if not os.path.isfile(file_path):
        raise ValueError(f"Input file not found: {file_path}")
    if not file_path.endswith(global_params["FILE_EXTENSION"]):
        raise ValueError(f"Expected a {global_params['FILE_EXTENSION']} file, got: {file_path}")
    if gt_file is not None and not os.path.isfile(gt_file):
        raise ValueError(f"GT file not found: {gt_file}")

    os.makedirs(output_dir, exist_ok=True)

    channel_num = global_params["channel_num"]
    num_output_slices = global_params["NUM_OUTPUT_SLICES"]
    norm_type = global_params["NORM"]
    eval_bs = global_params["INFERENCE_BATCH_SIZE"]
    ph_harm = global_params["PHASOR_HARMONICS"]

    print("Processing every pixel (whole image mode)")

    all_pred_list: List[np.ndarray] = []
    all_true_list: List[np.ndarray] = []

    print(f"\nProcessing {file_path}")
    print(f"  [1/6] Loading and preparing tensors...")
    t0 = time.time()
    x, y_true, w, spatial_shape = _prepare_file_tensors(
        file_path,
        repeats,
        channel_num,
        norm_type,
        wavelength=wavelength,
        gt_file_path=gt_file,
    )
    print(f"        Done in {time.time() - t0:.1f}s  |  pixels: {x.shape[0]}  shape: {spatial_shape}")

    print(f"  [2/6] Running inference...")
    t0 = time.time()
    y_pred = _batched_infer(model, device, x, w, eval_bs)
    elapsed_infer = time.time() - t0
    print(f"        Done in {elapsed_infer:.1f}s  ({x.shape[0] / elapsed_infer:.0f} pixels/s)")

    print(f"  [3/6] Computing metrics...")
    t0 = time.time()
    if y_true is not None:
        y_pred_orig, y_true_orig = invert_scale(y_pred, y_true, norm_type)
        mape = float(compute_mape(y_true_orig, y_pred_orig))
        mean_smape, median_smape = compute_smape(y_true_orig, y_pred_orig)
        rmse = float(np.sqrt(np.mean((y_true_orig - y_pred_orig) ** 2)))
        with torch.no_grad():
            yt = torch.tensor(y_true_orig, dtype=torch.float32)
            yp = torch.tensor(y_pred_orig, dtype=torch.float32)
            ph1 = float(phasor_loss(yp, yt, harmonic=ph_harm[0]).item())
            ph2 = float(phasor_loss(yp, yt, harmonic=ph_harm[1]).item())
        print(f"        Done in {time.time() - t0:.1f}s  |  MAPE={mape:.2f}%  SMAPE={mean_smape:.2f}%  RMSE={rmse:.1f}  ph1={ph1:.4f}  ph2={ph2:.4f}")
    else:
        y_pred_orig, _ = invert_scale(y_pred, np.zeros_like(y_pred), norm_type)
        y_true_orig = None
        mape = mean_smape = median_smape = rmse = ph1 = ph2 = None
        print(f"        No GT file — skipping metrics  ({time.time() - t0:.1f}s)")

    all_pred_list.append(y_pred_orig)
    if y_true_orig is not None:
        all_true_list.append(y_true_orig)

    print(f"  [4/6] Saving metrics.json...")
    t0 = time.time()
    block_dir = output_dir

    if y_true_orig is not None:
        file_metrics_json = {
            "mape": float(mape),
            "mean_smape": float(mean_smape),
            "median_smape": float(median_smape),
            "rmse": float(rmse),
            "phasor_h1": float(ph1),
            "phasor_h2": float(ph2),
        }
        with open(os.path.join(block_dir, "metrics.json"), "w") as jf:
            json.dump(file_metrics_json, jf, indent=2)
    else:
        print("        No GT — skipping metrics.json")
    print(f"        Done in {time.time() - t0:.1f}s")

    try:
        print(f"  [5/6] Saving prediction LSM and per-pixel maps...")
        t0 = time.time()
        H, W = spatial_shape
        pred_img_32 = y_pred_orig.T.reshape(global_params["OUTPUT_CHANNELS"], H, W)
        gt_img_32 = y_true_orig.T.reshape(global_params["OUTPUT_CHANNELS"], H, W) if y_true_orig is not None else None

        # Save prediction as LSM (32, H, W)
        tifffile.imwrite(os.path.join(block_dir, "pred32.lsm"), pred_img_32)

        # GT-based pixel maps (only if GT is available)
        if gt_img_32 is not None:
            ph2_map = _compute_phasor_h2_map(gt_img_32, pred_img_32, harmonic=global_params["PHASOR_HARMONICS"][1])
        else:
            ph2_map = None

        # 50 random spectra plots (always generated; input shown as step/box plot)
        spectra_dir = os.path.join(block_dir, "spectra")
        os.makedirs(spectra_dir, exist_ok=True)
        rng_spec = np.random.default_rng(31)
        coords = list(zip(rng_spec.integers(0, H, 50), rng_spec.integers(0, W, 50)))
        channel_num_local = global_params["channel_num"]
        group_size = global_params["OUTPUT_CHANNELS"] // channel_num_local
        bin_edges = np.arange(channel_num_local + 1) * group_size

        for i, (hh, ww) in enumerate(coords, start=1):
            pixel_idx = hh * W + ww
            pr_spec = pred_img_32[:, hh, ww]
            # Input: invert normalization then mean across repeats → (channel_num,)
            inp_vals = invert_scale(x[pixel_idx, 0, :, :], norm_type=norm_type).mean(axis=-1)

            fig, ax = plt.subplots(figsize=(9, 5))

            # Input as step/box plot on the same axis
            ax.stairs(inp_vals, bin_edges, fill=True, alpha=0.2, color='green', label='Input')
            ax.stairs(inp_vals, bin_edges, color='green', linewidth=1.5)

            # Prediction and optional GT
            ax.plot(range(global_params["OUTPUT_CHANNELS"]), pr_spec, 'r--', label='Pred', linewidth=2)
            if gt_img_32 is not None:
                gt_spec = gt_img_32[:, hh, ww]
                spec_eps = 1e-8
                spec_rmse = float(np.sqrt(np.mean((pr_spec - gt_spec) ** 2)))
                spec_smape = float(np.mean(2 * np.abs(pr_spec - gt_spec) / (np.abs(gt_spec) + np.abs(pr_spec) + spec_eps)) * 100)
                spec_ph2 = float(ph2_map[hh, ww])
                ax.plot(range(global_params["OUTPUT_CHANNELS"]), gt_spec, 'b-', label='GT', linewidth=2)
                ax.set_title(f'Spectrum ({hh},{ww})\nRMSE={spec_rmse:.1f}  SMAPE={spec_smape:.1f}%  Phasor h2={spec_ph2:.4f}')
            else:
                ax.set_title(f'Spectrum ({hh},{ww})')

            ax.set_xlabel('Output Channel')
            ax.set_ylabel('Intensity')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            plt.tight_layout()
            plt.savefig(os.path.join(spectra_dir, f"spectrum_{i:02d}_{hh}_{ww}.png"), dpi=130)
            plt.close(fig)

        print(f"        Done in {time.time() - t0:.1f}s")

        # Per-channel images
        print(f"  [6/6] Saving channel images (32 channels)...")
        t0 = time.time()
        channels_dir = os.path.join(block_dir, "channels")
        os.makedirs(channels_dir, exist_ok=True)
        eps = 1e-10

        for ch in range(global_params["OUTPUT_CHANNELS"]):
            pr_ch = np.nan_to_num(pred_img_32[ch], nan=0.0, posinf=0.0, neginf=0.0)

            if gt_img_32 is not None:
                gt_ch = np.nan_to_num(gt_img_32[ch], nan=0.0, posinf=0.0, neginf=0.0)
                smape_ch = np.clip(2 * np.abs(pr_ch - gt_ch) / (np.abs(gt_ch) + np.abs(pr_ch) + eps) * 100.0, 0, 50)
                rmse_ch = np.sqrt((pr_ch - gt_ch) ** 2)
                panels = [gt_ch, pr_ch, smape_ch, rmse_ch, ph2_map]
                titles = [f'GT ch {ch+1}', f'Pred ch {ch+1}', 'SMAPE (%)', 'RMSE', 'Phasor h2']
                cmaps  = ['gray', 'gray', 'magma', 'magma', 'viridis']
                vmins  = [None, None, 0, 0, 0]
                vmaxs  = [None, None, 50, np.max(pr_ch) / 5.0, 0.15]
            else:
                panels = [pr_ch]
                titles = [f'Pred ch {ch+1}']
                cmaps  = ['gray']
                vmins  = [None]
                vmaxs  = [None]

            n_cols = len(panels)
            fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
            if n_cols == 1:
                axes = [axes]

            for ax, panel, title, cmap, vmin, vmax in zip(axes, panels, titles, cmaps, vmins, vmaxs):
                im = ax.imshow(panel, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(title, fontsize=10)
                ax.axis('off')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig(os.path.join(channels_dir, f"channel_{ch+1:02d}.png"), dpi=140, bbox_inches='tight')
            plt.close(fig)
        print(f"        Done in {time.time() - t0:.1f}s")

    except Exception as e:
        print(f"Warning: outputs failed for {file_path}: {e}")

    if mape is not None:
        return {"mape": mape, "mean_smape": mean_smape, "median_smape": median_smape, "rmse": rmse, "ph1": ph1, "ph2": ph2}
    return {}


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on a single .lsm file")
    parser.add_argument("--repeats", type=int, required=True, help="Number of repeats in the input file (e.g. 3 → uses indices 0,1,2)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model .pt file")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the single .lsm file to evaluate")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder where all outputs will be written")
    parser.add_argument("--wavelength", type=float, required=True,
                        help="Excitation wavelength in nm (e.g. 405, 458, 488, 514, 561, 591, 633)")
    parser.add_argument("--gt_file", type=str, default=None,
                        help="(Optional) Path to the paired 32-channel GT .lsm file. If omitted, only the prediction is saved (no metrics).")
    parser.add_argument("--seed", type=int, default=global_params["SEED"], help="Random seed")
    parser.add_argument("--channels", type=int, default=None,
                        help="Number of input channels (overrides global_val.txt, must divide 32, e.g., 1 2 4 8 16 32)")
    args = parser.parse_args()

    # Override channel_num from command line if provided
    if args.channels is not None:
        valid = global_params["VALID_CHANNEL_NUMS"]
        if args.channels not in valid:
            raise ValueError(f"--channels must be one of {valid}")
        global_params["channel_num"] = args.channels
        global_params["GROUP_SIZE"] = global_params["GROUP_SIZE_DIVISOR"] // args.channels
        print(f"Overriding channel_num to {args.channels} from command line")

    set_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    repeats = list(range(args.repeats))
    model = FilmEarlyNet(replicate_count=args.repeats)

    try:
        ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.model_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)

    initialize_directories()

    metrics = evaluate_on_file(
        model=model,
        device=device,
        repeats=repeats,
        file_path=args.input_file,
        output_dir=args.output_dir,
        wavelength=args.wavelength,
        gt_file=args.gt_file,
    )

    if metrics:
        print(f"Metrics -> MAPE: {metrics['mape']:.4f}% | Mean_SMAPE: {metrics['mean_smape']:.4f}% | Median_SMAPE: {metrics['median_smape']:.4f}% | RMSE: {metrics['rmse']:.6f} | Phasor h1: {metrics['ph1']:.6f} | Phasor h2: {metrics['ph2']:.6f}")
    else:
        print("Inference complete. No GT provided — metrics not computed.")


if __name__ == "__main__":
    main()


