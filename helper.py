import os
import random
import re

import numpy as np
import torch

from config import load_global_params

global_params = load_global_params()


def set_seeds(seed=None):
    """Seed Python, NumPy and PyTorch for reproducibility."""
    if seed is None:
        seed = global_params["SEED"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seeds set to {seed} for reproducibility")


def get_device(cuda_index=0):
    """Pick the best available device: CUDA GPU, then Apple GPU (MPS), then CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{cuda_index}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def initialize_directories():
    """Create PLOTS_DIR if needed and return its path."""
    plots_dir = global_params["PLOTS_DIR"]
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def normalize_data(x, y=None, norm_type=None):
    """Apply log1p to x (and y) when norm_type is 'log'; otherwise return them unchanged."""
    if norm_type is None:
        norm_type = global_params["NORM"]
    if norm_type == 'log':
        x = np.log1p(x)
        if y is not None:
            y = np.log1p(y)
    return (x, y) if y is not None else x


def invert_scale(pred, true=None, norm_type=None):
    """Undo normalize_data (expm1 when norm_type is 'log')."""
    if norm_type is None:
        norm_type = global_params["NORM"]
    if norm_type == 'log':
        pred = np.expm1(pred)
        if true is not None:
            true = np.expm1(true)
    return (pred, true) if true is not None else pred


_BLOCK_RE = re.compile(r"Block[\s_-]*(\d+)", re.IGNORECASE)


def file_to_block_number(filepath):
    """Return N from the last 'Block N' / 'Block_N' / 'BlockN' token in the path, or None.

    Works whether or not "Block" is a separate word, e.g.
    'Image 11 Block 5.lsm' -> 5, 'Image_11_Block_1.lsm' -> 1, 'pos4_Image12Block_3.lsm' -> 3.
    """
    matches = _BLOCK_RE.findall(os.path.splitext(filepath)[0])
    return int(matches[-1]) if matches else None


def file_to_wavelength(filepath):
    """Return the excitation wavelength (nm) for a file, based on its block number.

    Falls back to BLOCK_NUMBER_DEFAULT (with a warning) if no block number is found.
    """
    wavelength_map = global_params["WAVELENGTH_MAP"]
    block_num = file_to_block_number(filepath)
    if block_num is None or block_num not in wavelength_map:
        block_num = global_params["BLOCK_NUMBER_DEFAULT"]
        print(f"Warning: Could not parse block number from {filepath}, defaulting to {block_num}")
    return float(wavelength_map[block_num])


def compute_mape(y_true, y_pred):
    """Mean absolute percentage error (%), ignoring entries where y_true is ~0."""
    eps = global_params["MAPE_EPSILON"]
    y_true_f, y_pred_f = y_true.flatten(), y_pred.flatten()
    nonzero_mask = np.abs(y_true_f) > eps
    if not np.any(nonzero_mask):
        return 0.0
    return np.mean(np.abs((y_true_f[nonzero_mask] - y_pred_f[nonzero_mask]) / y_true_f[nonzero_mask])) * 100.0


def compute_smape(y_true, y_pred):
    """Symmetric MAPE (%) per spectrum; returns (mean, median) over spectra."""
    eps = global_params["MAPE_EPSILON"]
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
        y_pred = y_pred.reshape(1, -1)
    per_sample_smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps), axis=1) * 100.0
    return float(np.mean(per_sample_smape)), float(np.median(per_sample_smape))
