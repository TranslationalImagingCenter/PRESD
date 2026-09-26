import os
import numpy as np
import random
import torch
from config import load_global_params

# Load global parameters
global_params = load_global_params()

# -------------------------------------------------------------------
# Random seed setup
# -------------------------------------------------------------------
def set_seeds(seed=None):
    """Set random seeds for reproducibility across all libraries."""
    if seed is None:
        seed = global_params["DEFAULT_HELPER_SEED"]
        
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seeds set to {seed} for reproducibility")

# -------------------------------------------------------------------
# Directory and dataset management
# -------------------------------------------------------------------
def get_cache_dir():
    """Get cache directory."""
    return global_params["CACHE_DIR"]

def get_training_folders():
    """Get training folders."""
    return global_params["TRAIN_FOLDERS"]

def get_plots_dir():
    """Get plots directory."""
    return global_params["PLOTS_DIR"]

def initialize_directories():
    """Initialize plots directory."""
    plots_dir = get_plots_dir()
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

# -------------------------------------------------------------------
# Cache management
# -------------------------------------------------------------------
def get_cache_filename(cache_key, max_samples, global_params_local):
    """Generate a unique filename for the cache based on parameters."""
    cache_dir = get_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    
    channel_num = global_params_local.get("channel_num", global_params["channel_num"])
    
    # Get repeats/position_list for cache filename - prefer explicit REPEATS key
    repeats = global_params_local.get("REPEATS", global_params_local.get("POSITION_LIST", [1, 2, 3]))
    pos_str = "-".join(map(str, repeats))
    
    return os.path.join(cache_dir, f"cache_{cache_key}_{max_samples}_ch{channel_num}_pos{pos_str}.npz")

def save_to_cache(data, cache_key, max_samples, global_params_local):
    """Save processed data to cache."""
    cache_file = get_cache_filename(cache_key, max_samples, global_params_local)
    x_train, w_train, y_train, x_val, w_val, y_val = data
    
    np.savez_compressed(cache_file, x_train=x_train, w_train=w_train, y_train=y_train,
                       x_val=x_val, w_val=w_val, y_val=y_val)
    print(f"Data saved to cache: {cache_file}")

def load_from_cache(cache_key, max_samples, global_params_local):
    """Try to load data from cache."""
    cache_file = get_cache_filename(cache_key, max_samples, global_params_local)
    
    if not os.path.exists(cache_file):
        print(f"Cache file not found: {cache_file}")
        return None
    
    try:
        print(f"Loading data from cache: {cache_file}")
        data = np.load(cache_file)
        print(f"Cache loaded successfully!")
        return data['x_train'], data['w_train'], data['y_train'], data['x_val'], data['w_val'], data['y_val']
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None

# -------------------------------------------------------------------
# Normalization helpers
# -------------------------------------------------------------------
def normalize_data(x, y=None, norm_type=None):
    """Apply normalization to input and output data."""
    if norm_type is None:
        norm_type = global_params["NORM"]
        
    if norm_type == 'log':
        x = np.log1p(x)
        if y is not None:
            y = np.log1p(y)
    return (x, y) if y is not None else x

def invert_scale(pred, true=None, norm_type=None):
    """Invert normalization for predictions and ground truth."""
    if norm_type is None:
        norm_type = global_params["NORM"]
        
    if norm_type == 'log':
        pred = np.expm1(pred)
        if true is not None:
            true = np.expm1(true)
    return (pred, true) if true is not None else pred

def file_to_wavelength_norm(filepath):
    """Maps a file path to a normalized wavelength value by extracting the Block number.
    Searches the full path (including parent directories) for 'Block N'."""
    wavelength_map = global_params["WAVELENGTH_MAP"]

    try:
        # Strip extension then split on spaces, slashes, and underscores to find "Block N"
        path_no_ext = os.path.splitext(filepath)[0]
        parts = path_no_ext.replace(os.sep, ' ').replace('/', ' ').replace('_', ' ').split()
        if "Block" in parts:
            block_idx = parts.index("Block") + 1
            block_num = int(parts[block_idx])
        else:
            block_num = global_params["BLOCK_NUMBER_DEFAULT"]
            print(f"Warning: Could not parse block number from {filepath}, defaulting to {block_num}")
    except (ValueError, IndexError):
        block_num = global_params["BLOCK_NUMBER_DEFAULT"]
        print(f"Warning: Could not parse block number from {filepath}, defaulting to {block_num}")

    return wavelength_map.get(block_num, wavelength_map[1]) / 1.0

def compute_mape(y_true, y_pred):
    """Compute Mean Absolute Percentage Error."""
    eps = global_params["MAPE_EPSILON"]
    y_true_f, y_pred_f = y_true.flatten(), y_pred.flatten()
    nonzero_mask = (np.abs(y_true_f) > eps)
    
    if not np.any(nonzero_mask):
        return 0.0
        
    return np.mean(np.abs((y_true_f[nonzero_mask] - y_pred_f[nonzero_mask]) / y_true_f[nonzero_mask])) * 100.0


def compute_smape(y_true, y_pred):
    """Compute Symmetric Mean Absolute Percentage Error.
    Returns tuple: (mean_smape, median_smape) as percentages."""
    eps = global_params["MAPE_EPSILON"]
    
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
        y_pred = y_pred.reshape(1, -1)
    
    per_sample_smape = []
    for i in range(y_true.shape[0]):
        gt = y_true[i]
        pred = y_pred[i]
        denom = np.abs(gt) + np.abs(pred) + eps
        smape = np.mean(2 * np.abs(pred - gt) / denom) * 100.0
        per_sample_smape.append(smape)
    
    per_sample_smape = np.array(per_sample_smape)
    return float(np.mean(per_sample_smape)), float(np.median(per_sample_smape)) 