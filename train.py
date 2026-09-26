import os
import numpy as np
import time
import tifffile
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import hashlib
import argparse
import json
from multiprocessing import Pool, cpu_count
import functools

from config import load_global_params
from NN import FilmEarlyNet, combined_spectral_loss, phasor_loss
from helper import (
    normalize_data, invert_scale, file_to_wavelength_norm,
    compute_mape, compute_smape, set_seeds, get_training_folders,
    get_plots_dir, initialize_directories,
)
from plot import plot_comparison

# -------------------------------------------------------------------
# Load global parameters
# -------------------------------------------------------------------
global_params = load_global_params()
TEST_TYPE = global_params["TEST_TYPE"]
channel_num = global_params["channel_num"]
training_ratio = global_params["training_ratio"]
BATCHSIZE = global_params["BATCHSIZE"]
EPOCHS = global_params["EPOCHS"]
NORM = global_params["NORM"]
MAX_SAMPLES = global_params["MAX_SAMPLES"]
# REPEATS will be set from command line arguments

# Validate channel_num - must be a divisor of 32
valid_channel_nums = global_params["VALID_CHANNEL_NUMS"]
if channel_num not in valid_channel_nums:
    raise ValueError(f"channel_num must be one of {valid_channel_nums} (32 must be divisible by channel_num)")
print(f"Using channel_num = {channel_num} ({global_params['OUTPUT_CHANNELS']}/{channel_num} = {global_params['OUTPUT_CHANNELS']//channel_num})")

# Set random seeds for reproducibility
SEED = global_params["SEED"]
set_seeds(SEED)

# INPUT_FOLDERS: directories containing the input rep_N.lsm files (e.g. 4ch/ folders).
# GT_FOLDERS: paired directories each containing a ground_truth.lsm file.
INPUT_FOLDERS = []
GT_FOLDERS = []

# -------------------------------------------------------------------
# Parallel file processing worker function
# -------------------------------------------------------------------
def process_single_file(args):
    """
    Worker function to process a single file in parallel.
    This function will be called by multiple processes.
    
    Args:
        args: Tuple containing (folder, filename, block_number, file_index, samples_per_file)
    
    Returns:
        Tuple of (x_data, y_data, w_data) or None if file processing failed
    """
    folder, f, gt_folder, block_number, file_index, samples_per_file = args

    try:
        f_path = os.path.join(folder, f)

        # Read input file — natively has channel_num channels
        # Actual file shape: (num_repeats, channel_num, H, W)
        data = tifffile.TiffFile(f_path).asarray()
        wave_norm = file_to_wavelength_norm(f_path)

        # Use process-specific random seed for reproducibility
        np.random.seed(SEED + file_index)

        num_repeats_in_file = data.shape[0]
        M = data.shape[-2] * data.shape[-1]  # H * W

        # Build input: data[i] gives (channel_num, H, W) for repeat i
        pos_list = REPEATS
        pos_count = len(pos_list)
        raw_in = []
        for i in pos_list:
            if i >= num_repeats_in_file:
                break
            raw_in.append(data[i].reshape(channel_num, -1))  # (channel_num, M)
        if len(raw_in) < pos_count:
            print(f"    [Process] Not enough repeats (has {num_repeats_in_file}) for input, skipping {f}.")
            return None

        compressed_in = np.zeros((channel_num, len(pos_list), M))
        for p_idx, slice_ch in enumerate(raw_in):
            compressed_in[:, p_idx] = slice_ch

        # Load 32-channel ground truth — same filename as input file, in gt_folder
        # GT shape: (32, H, W)
        gt_path = os.path.join(gt_folder, f)
        if not os.path.isfile(gt_path):
            print(f"    [Process] GT file not found, skipping {f}: {gt_path}")
            return None
        gt_data = tifffile.TiffFile(gt_path).asarray()
        out_32 = gt_data.reshape(global_params["OUTPUT_CHANNELS"], -1)  # (32, M)

        # MEMORY EFFICIENT: Sample a subset of points from this file
        if M > samples_per_file:
            idx = np.random.choice(M, samples_per_file, replace=False)
            file_in = compressed_in[:, :, idx]
            file_out = out_32[:, idx]
            file_wave = np.full((samples_per_file,), wave_norm, dtype=np.float32)
        else:
            file_in = compressed_in
            file_out = out_32
            file_wave = np.full((M,), wave_norm, dtype=np.float32)
        
        # Normalize each file's data individually
        file_in, file_out = normalize_data(file_in, file_out, NORM)
        
        # Reshape for network
        file_samples = file_in.shape[2]
        x_data = np.transpose(file_in, (2, 0, 1))  # (samples,16,3)
        x_data = np.expand_dims(x_data, axis=1)    # (samples,1,16,3)
        y_data = file_out.T                        # (samples,32)
        
        print(f"    [Process] Processed {f}: shape={data.shape}, wave_norm={wave_norm:.2f}, samples={file_samples}")
        
        return (x_data, y_data, file_wave)
        
    except Exception as e:
        print(f"    [Process] Error processing {f}: {e}")
        return None

# -------------------------------------------------------------------
# Parallel data loading function
# -------------------------------------------------------------------
def load_data(selected_block=None):
    """
    Load and process training data from LSM files in parallel.

    Args:
        selected_block: If provided, only load data from this specific block
    """
    initialize_directories()

    # Collect all valid (input_file, gt_folder) pairs from INPUT_FOLDERS / GT_FOLDERS
    valid_files = []
    for folder_idx, folder in enumerate(INPUT_FOLDERS):
        print(folder)
        if not os.path.isdir(folder):
            print(f"Input folder not found: {folder}")
            continue

        gt_folder = GT_FOLDERS[folder_idx] if folder_idx < len(GT_FOLDERS) else GT_FOLDERS[0]

        files = sorted([f for f in os.listdir(folder) if f.endswith(global_params["FILE_EXTENSION"])])
        for f_idx, f in enumerate(files):
            block_number = f_idx + 1
            if selected_block is not None and block_number != selected_block:
                continue
            valid_files.append((folder, f, gt_folder, block_number))

    total_valid_files = len(valid_files)
    if total_valid_files == 0:
        raise ValueError("No valid files found.")

    print(f"\nFound {total_valid_files} valid files.")
    samples_per_file = MAX_SAMPLES // total_valid_files
    print(f"Taking approximately {samples_per_file} samples per file.")

    num_processes = max(1, min(int(cpu_count() * 0.75), total_valid_files))
    print(f"Using {num_processes} parallel processes (out of {cpu_count()} available CPUs)")

    process_args = []
    for i, (folder, f, gt_folder, block_number) in enumerate(valid_files):
        process_args.append((folder, f, gt_folder, block_number, i + 1, samples_per_file))
    
    # Process files in parallel
    print("\nProcessing files in parallel...")
    start_time = time.time()
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_single_file, process_args)
    
    processing_time = time.time() - start_time
    print(f"Parallel processing completed in {processing_time:.2f} seconds")
    
    # Filter out failed files and collect results
    successful_results = [r for r in results if r is not None]
    total_files_read = len(successful_results)
    
    print(f"Successfully processed {total_files_read} out of {total_valid_files} files")
    
    if not successful_results:
        raise ValueError("No valid data found after parallel processing.")

    # Combine the processed, normalized, and already-sampled data
    print("Combining results from parallel processing...")
    input_list = [r[0] for r in successful_results]
    output_list = [r[1] for r in successful_results]
    wave_list = [r[2] for r in successful_results]
    
    x_data = np.concatenate(input_list, axis=0)
    y_data = np.concatenate(output_list, axis=0)
    w_data = np.concatenate(wave_list, axis=0)

    total_samples = x_data.shape[0]
    print(f"\nTotal collected samples: {total_samples}")

    # Train/val split
    np.random.seed(SEED)  # Set seed for reproducible split
    inds = np.random.permutation(total_samples)
    n_train = int(training_ratio * total_samples)
    train_idx, val_idx = inds[:n_train], inds[n_train:]
    x_train, x_val = x_data[train_idx], x_data[val_idx]
    y_train, y_val = y_data[train_idx], y_data[val_idx]
    w_train, w_val = w_data[train_idx], w_data[val_idx]

    print(f"\nTrain set => x:{x_train.shape}, y:{y_train.shape}, wave:{w_train.shape}")
    print(f"Val set   => x:{x_val.shape}, y:{y_val.shape}, wave:{w_val.shape}")

    return (x_train, w_train, y_train, x_val, w_val, y_val)

def _create_model_state(model, epoch, val_loss, metrics):
    """Helper to create model state dictionary."""
    return {
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
        'metrics': metrics
    }

def _compute_epoch_metrics(model, x_train_t, w_train_t, y_train_t, x_val_t, w_val_t, y_val_t, loss_fn, norm_type):
    """Compute all metrics for current epoch."""
    model.eval()
    with torch.no_grad():
        # Training predictions (batched to avoid OOM)
        train_preds = []
        eval_batch_size = global_params["EVAL_BATCH_SIZE"]
        n_samples = x_train_t.shape[0]
        
        for start_idx in range(0, n_samples, eval_batch_size):
            end_idx = min(start_idx + eval_batch_size, n_samples)
            batch_pred = model(x_train_t[start_idx:end_idx], w_train_t[start_idx:end_idx])
            train_preds.append(batch_pred.cpu().numpy())
        
        train_pred = np.vstack(train_preds)
        train_true = y_train_t.cpu().numpy()
        
        # Validation predictions (batched to avoid OOM)
        val_preds = []
        n_val_samples = x_val_t.shape[0]
        val_loss_accum = 0.0
        phasor_val_accum = 0.0

        for start_idx in range(0, n_val_samples, eval_batch_size):
            end_idx = min(start_idx + eval_batch_size, n_val_samples)
            batch_x = x_val_t[start_idx:end_idx]
            batch_w = w_val_t[start_idx:end_idx]
            batch_y = y_val_t[start_idx:end_idx]
            batch_pred = model(batch_x, batch_w)
            val_preds.append(batch_pred.cpu().numpy())
            val_loss_accum += loss_fn(batch_pred, batch_y, batch_x).item() * (end_idx - start_idx)
            phasor_val_accum += phasor_loss(batch_pred, batch_y, harmonic=global_params["DEFAULT_HARMONIC"]).item() * (end_idx - start_idx)

        val_pred_np = np.vstack(val_preds)
        val_true_np = y_val_t.cpu().numpy()
        val_loss = val_loss_accum / n_val_samples
        phasor_val = phasor_val_accum / n_val_samples
        
        # MAPE calculations - invert normalization to get original scale
        train_pred_orig, train_true_orig = invert_scale(train_pred, train_true, norm_type)
        val_pred_orig, val_true_orig = invert_scale(val_pred_np, val_true_np, norm_type)
        
        # Phasor metrics for both harmonics - calculated on original scale data for reporting
        phasor_harmonics = global_params["PHASOR_HARMONICS"]
        
        # Convert to tensors for phasor calculation on original scale
        val_pred_orig_tensor = torch.tensor(val_pred_orig, dtype=torch.float32)
        val_true_orig_tensor = torch.tensor(val_true_orig, dtype=torch.float32)
        
        val_ph_h1 = phasor_loss(val_pred_orig_tensor, val_true_orig_tensor, harmonic=phasor_harmonics[0]).item()
        val_ph_h2 = phasor_loss(val_pred_orig_tensor, val_true_orig_tensor, harmonic=phasor_harmonics[1]).item()
        
        # Training phasor - reuse already-computed batched training predictions
        sample_pred_orig, sample_true_orig = train_pred_orig, train_true_orig
        
        # Convert to tensors for phasor calculation
        sample_pred_orig_tensor = torch.tensor(sample_pred_orig, dtype=torch.float32)
        sample_true_orig_tensor = torch.tensor(sample_true_orig, dtype=torch.float32)
        
        train_ph_h1 = phasor_loss(sample_pred_orig_tensor, sample_true_orig_tensor, harmonic=phasor_harmonics[0]).item()
        train_ph_h2 = phasor_loss(sample_pred_orig_tensor, sample_true_orig_tensor, harmonic=phasor_harmonics[1]).item()
        
        train_mape = compute_mape(train_true_orig, train_pred_orig)
        val_mape = compute_mape(val_true_orig, val_pred_orig)
        
        train_mean_smape, train_median_smape = compute_smape(train_true_orig, train_pred_orig)
        val_mean_smape, val_median_smape = compute_smape(val_true_orig, val_pred_orig)
        
        torch.cuda.empty_cache()
        
        return {
            'val_loss': val_loss,
            'phasor_val': phasor_val,
            'train_mape': train_mape,
            'val_mape': val_mape,
            'train_mean_smape': train_mean_smape,
            'val_mean_smape': val_mean_smape,
            'train_median_smape': train_median_smape,
            'val_median_smape': val_median_smape,
            'train_phasor_h1': train_ph_h1,
            'val_phasor_h1': val_ph_h1,
            'train_phasor_h2': train_ph_h2,
            'val_phasor_h2': val_ph_h2
        }

def train_model(model, x_train, w_train, y_train, x_val, w_val, y_val, lambda_phasor=None, harmonic=None, epochs=None, batch_size=None):
    """Train the FilmEarlyNet model on the provided data using combined MSE + Phasor loss."""
    start_time = time.time()
    
    # Use global parameters with optional overrides
    if lambda_phasor is None:
        lambda_phasor = global_params["LAMBDA_PHASOR"]
    if harmonic is None:
        harmonic = global_params["DEFAULT_HARMONIC"]
    if epochs is None:
        epochs = EPOCHS
    if batch_size is None:
        batch_size = BATCHSIZE
    
    # Setup device and model
    default_gpus = global_params["DEFAULT_GPUS"]
    device = torch.device(f"cuda:{default_gpus[0]}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if torch.cuda.is_available():
        print(f"Using GPUs {' & '.join(map(str, default_gpus))} for training")
        
        # Show detailed GPU information
        print("GPU Status:")
        for gpu_id in default_gpus:
            gpu_name = torch.cuda.get_device_name(gpu_id)
            gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
            print(f"  GPU {gpu_id}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        model = nn.DataParallel(model, device_ids=default_gpus)
        
        # Show memory allocation after model setup
        print("GPU Memory After Model Setup:")
        for gpu_id in default_gpus:
            allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
            cached = torch.cuda.memory_reserved(gpu_id) / 1e9
            print(f"  GPU {gpu_id}: {allocated:.2f} GB allocated, {cached:.2f} GB cached")
    
    # Setup optimizer and scheduler
    initial_lr = global_params["INITIAL_LR"]
    weight_decay = global_params["WEIGHT_DECAY"]
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=global_params["LR_FACTOR"], 
        patience=global_params["LR_PATIENCE"], verbose=True, 
        min_lr=global_params["MIN_LR"]
    )
    
    # Convert data to tensors
    x_train_t = torch.tensor(x_train, dtype=torch.float32).to(device)
    w_train_t = torch.tensor(w_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).to(device)
    w_val_t = torch.tensor(w_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    
    # Setup loss function - always use combined MSE + Phasor loss
    lambda_consistency = global_params["LAMBDA_CONSISTENCY"]
    print(f"Training with combined MSE + Phasor loss (lambda_phasor={lambda_phasor}, harmonic={harmonic})")
    loss_fn = lambda pred, target, input_x: combined_spectral_loss(
        pred, target, input_x, lambda_consistency, lambda_phasor, harmonic
    )
    
    # Initialize tracking
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stop_patience = global_params["EARLY_STOP_PATIENCE"]
    
    metrics = {
        'train_losses': [], 'val_losses': [], 'train_mapes': [], 'val_mapes': [],
        'train_mean_smapes': [], 'val_mean_smapes': [], 'train_median_smapes': [], 'val_median_smapes': [],
        'phasor_values': [], 'train_phasor_h1': [], 'val_phasor_h1': [],
        'train_phasor_h2': [], 'val_phasor_h2': []
    }
    
    n_samples = x_train_t.shape[0]
    warmup_epochs = global_params["WARMUP_EPOCHS"]
    grad_clip_max_norm = global_params["GRAD_CLIP_MAX_NORM"]
    
    # Training loop
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        perm = np.random.permutation(n_samples)
        batch_losses = []
        
        # Learning rate warmup
        lr_multiplier = (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = initial_lr * lr_multiplier
        
        # Train on batches
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            idx = perm[start_idx:end_idx]
            
            batch_x = x_train_t[idx]
            batch_w = w_train_t[idx]
            batch_y = y_train_t[idx]
            
            optimizer.zero_grad()
            pred = model(batch_x, batch_w)  # Pass wavelength information to FilmEarlyNet
            loss = loss_fn(pred, batch_y, batch_x)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            optimizer.step()
            batch_losses.append(loss.item())
        
        # Compute epoch metrics
        epoch_train_loss = np.mean(batch_losses)
        metrics['train_losses'].append(epoch_train_loss)
        
        epoch_metrics = _compute_epoch_metrics(model, x_train_t, w_train_t, y_train_t, x_val_t, w_val_t, y_val_t, loss_fn, NORM)
        
        # Update metrics
        for key in ['val_loss', 'train_mape', 'val_mape', 'train_mean_smape', 'val_mean_smape', 'train_median_smape', 'val_median_smape', 'train_phasor_h1', 'val_phasor_h1', 'train_phasor_h2', 'val_phasor_h2']:
            if key == 'val_loss':
                metrics['val_losses'].append(epoch_metrics[key])
            elif 'mape' in key or 'smape' in key:
                metrics[f"{key}s"].append(epoch_metrics[key])
            else:
                metrics[key].append(epoch_metrics[key])
        
        metrics['phasor_values'].append(epoch_metrics['phasor_val'])
        
        # Update learning rate and check early stopping
        scheduler.step(epoch_metrics['val_loss'])
        
        if epoch_metrics['val_loss'] < best_val_loss:
            best_val_loss = epoch_metrics['val_loss']
            best_model_state = _create_model_state(model, epoch, epoch_metrics['val_loss'], metrics)
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
        
        # Print progress
        epoch_time = time.time() - epoch_start_time
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {epoch_train_loss:.6f}, "
              f"Val Loss: {epoch_metrics['val_loss']:.6f}, "
              f"Train MAPE: {epoch_metrics['train_mape']:.2f}%, "
              f"Val MAPE: {epoch_metrics['val_mape']:.2f}%, "
              f"Val Mean_SMAPE: {epoch_metrics['val_mean_smape']:.2f}%, "
              f"Val Median_SMAPE: {epoch_metrics['val_median_smape']:.2f}%, "
              f"Time: {epoch_time:.2f}s")
        print(f"            Phasor(h={global_params['PHASOR_HARMONICS'][0]}): Train={epoch_metrics['train_phasor_h1']:.6f}, Val={epoch_metrics['val_phasor_h1']:.6f}")
        print(f"            Phasor(h={global_params['PHASOR_HARMONICS'][1]}): Train={epoch_metrics['train_phasor_h2']:.6f}, Val={epoch_metrics['val_phasor_h2']:.6f}, LR={lr:.6f}")
        
        # Show GPU memory usage every 5 epochs
        if torch.cuda.is_available() and (epoch + 1) % 5 == 0:
            print(f"            GPU Memory Usage (Epoch {epoch+1}):")
            for gpu_id in default_gpus:
                allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
                cached = torch.cuda.memory_reserved(gpu_id) / 1e9
                print(f"              GPU {gpu_id}: {allocated:.2f} GB allocated, {cached:.2f} GB cached")
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Final GPU memory summary
    if torch.cuda.is_available():
        print("Final GPU Memory Summary:")
        for gpu_id in default_gpus:
            allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
            cached = torch.cuda.memory_reserved(gpu_id) / 1e9
            max_allocated = torch.cuda.max_memory_allocated(gpu_id) / 1e9
            print(f"  GPU {gpu_id}: {allocated:.2f} GB allocated, {cached:.2f} GB cached, {max_allocated:.2f} GB peak")
    
    # Use final state if no early stopping occurred
    if best_model_state is None:
        best_model_state = _create_model_state(model, epochs - 1, metrics['val_losses'][-1], metrics)
    
    # Prepare results
    return {
        'model_state': best_model_state,
        'training_time': training_time,
        'best_epoch': best_model_state['epoch'],
        'best_val_loss': best_model_state['val_loss'],
        **{k: v for k, v in metrics.items()}
    }

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train models with different phasor loss weights (PARALLEL VERSION)")
    parser.add_argument('--harmonic', type=int, default=global_params["DEFAULT_HARMONIC"], 
                        help=f'Harmonic to use for phasor calculation (default: {global_params["DEFAULT_HARMONIC"]})')
    parser.add_argument('--block', type=str, default='all',
                        help='Block to train on: specific number or "all" (default: all)')
    parser.add_argument('--input_folders', type=str, nargs='+', required=True,
                        help='Directories containing the input rep_N.lsm files (e.g. .../4ch/), one per gt_folder')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Custom model name for saving (e.g., "NN_phasor_lambda_1.0_pos_1_2_3")')
    parser.add_argument('--processes', type=int, default=None,
                        help='Number of parallel processes to use (default: 75% of available CPUs)')
    parser.add_argument('--repeats', type=int, nargs='+', required=True,
                        help='Position repeats to use for training (REQUIRED, e.g., --repeats 1 2 3 5)')
    parser.add_argument('--channels', type=int, default=None,
                        help='Number of input channels (overrides global_val.txt, must divide 32, e.g., 1 2 4 8 16 32)')
    parser.add_argument('--gt_folders', type=str, nargs='+', required=True,
                        help='Directories containing the paired 32-channel GT LSM files (one per train folder, matched by filename)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Root directory for training outputs (overrides PLOTS_DIR in global_val.txt)')
    args = parser.parse_args()

    # Override channel_num from command line if provided
    if args.channels is not None:
        if args.channels not in global_params["VALID_CHANNEL_NUMS"]:
            raise ValueError(f"--channels must be one of {global_params['VALID_CHANNEL_NUMS']}")
        channel_num = args.channels
        global_params["channel_num"] = args.channels
        global_params["GROUP_SIZE"] = global_params["GROUP_SIZE_DIVISOR"] // args.channels
        print(f"Overriding channel_num to {channel_num} from command line")
    
    # Override number of processes if specified
    if args.processes:
        # Monkey patch cpu_count for the load_data function
        original_cpu_count = cpu_count
        cpu_count = lambda: args.processes
        print(f"Using {args.processes} processes as specified by --processes argument")
    
    print(f"PARALLEL TRAINING VERSION - Using {cpu_count()} available CPU cores")
    
    # Show CUDA/GPU availability
    print(f"\nGPU Information:")
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print(f"CUDA Available: No - Using CPU only")
    
    # Set position repeats from command line - REQUIRED
    if args.repeats:
        REPEATS = list(args.repeats)
        print(f"Using position repeats from command line: {REPEATS}")
    else:
        raise ValueError("ERROR: --repeats argument is required. Please specify position repeats explicitly (e.g., --repeats 1 2 3 5)")
    
    # Set input and GT folders (both required)
    INPUT_FOLDERS = args.input_folders
    GT_FOLDERS = args.gt_folders
    print(f"Input folders: {INPUT_FOLDERS}")
    print(f"GT folders:    {GT_FOLDERS}")
    
    # Convert block argument to the right format
    selected_block = None
    if args.block.lower() != 'all':
        try:
            selected_block = int(args.block)
            print(f"Training on block {selected_block} only")
        except ValueError:
            print(f"Invalid block value: {args.block}. Using all blocks.")
            selected_block = None
    else:
        print("Training on all blocks")
    
    # Different lambda_phasor values to try
    lambda_phasor_values = [global_params["LAMBDA_PHASOR_VALUES"]]
    
    # Print training settings
    print(f"\nTraining settings:")
    print(f"  Lambda phasor values: {lambda_phasor_values}")
    print(f"  Harmonic: {args.harmonic}")
    print(f"  Block: {args.block}")
    print(f"  Position repeats: {REPEATS}")
    print(f"  Input folders: {INPUT_FOLDERS}")
    
    # Build experiment directory name from input folder names and repeat indices
    folder_names = "_".join(os.path.basename(f.rstrip('/')) for f in INPUT_FOLDERS)
    rep_part = "_".join(map(str, REPEATS))
    experiment_folder_name = f"{folder_names}_rep_{rep_part}"

    if args.output_dir is not None:
        global_params["PLOTS_DIR"] = args.output_dir
    plots_dir = initialize_directories()
    train_base_dir = os.path.join(plots_dir, "train", f"{channel_num}_ch")
    os.makedirs(train_base_dir, exist_ok=True)
    experiment_dir = os.path.join(train_base_dir, experiment_folder_name)
    os.makedirs(experiment_dir, exist_ok=True)

    print(f"Experiment folder name: {experiment_folder_name}")
    print(f"Training results will be saved to: {experiment_dir}")
    
    # 1) Load data (PARALLEL VERSION)
    print("\n" + "="*60)
    print("STARTING PARALLEL DATA LOADING")
    print("="*60)
    data_start_time = time.time()
    
    x_train, w_train, y_train, x_val, w_val, y_val = load_data(selected_block)
    
    data_load_time = time.time() - data_start_time
    print("="*60)
    print(f"PARALLEL DATA LOADING COMPLETED IN {data_load_time:.2f} SECONDS")
    print("="*60)
    
    # 2) Print information about the dataset
    print(f"\nTraining set: {x_train.shape}, Validation set: {x_val.shape}")
    
    # 3) Create and train models with different lambda_phasor values
    results = {}
    
    # Train models with combined MSE + Phasor loss using different lambda values
    for lambda_value in lambda_phasor_values:
        # Include block info in model name if specific block is used
        block_info = f"_block{selected_block}" if selected_block is not None else ""
        model_name = f"Phasor_lambda_{lambda_value}{block_info}"
        print(f"\n===== Training Model: {model_name} =====")
        
        # Create model with correct replicate count for current REPEATS
        replicate_count = len(REPEATS)
        print(f"Creating model with REPEATS: {REPEATS} (replicate_count: {replicate_count})")
        
        phasor_model = FilmEarlyNet(replicate_count=replicate_count)
        phasor_results = train_model(
            phasor_model,
            x_train, w_train, y_train, 
            x_val, w_val, y_val, 
            lambda_phasor=lambda_value,
            harmonic=args.harmonic,
            epochs=EPOCHS, 
            batch_size=BATCHSIZE
        )
        results[model_name] = phasor_results

        # Save model with all metrics included
        # Use custom model name if provided, otherwise use default naming
        if args.model_name:
            model_filename = f"{args.model_name}_model.pt"
            metrics_filename = f"{args.model_name}_metrics.json"
        else:
            # Simple naming for compatibility with test scripts
            model_filename = "NN.pt"
            metrics_filename = f"{TEST_TYPE}_phasor_lambda_{lambda_value}{block_info}_metrics.json"
            
        model_path = os.path.join(experiment_dir, model_filename)
        torch.save(phasor_results['model_state'], model_path)
        print(f"Saved model to: {model_path}")

        # Also save metrics separately as JSON for easier access
        metrics_path = os.path.join(experiment_dir, metrics_filename)
        json_indent = global_params["JSON_INDENT"]
        with open(metrics_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            metrics = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in phasor_results.items()
                    if k != 'model_state'}
            json.dump(metrics, f, indent=json_indent)
        print(f"Saved metrics to: {metrics_path}")
    
    # 4) Compare models
    print("\n===== Lambda Phasor Value Comparison =====")
    for model_type, res in results.items():
        print(f"{model_type} performance:")
        print(f"  Best validation loss: {res['best_val_loss']:.6f}")
        print(f"  Training time: {res['training_time']:.2f} seconds")
        if len(res['val_mapes']) > 0:
            best_epoch = len(res['val_mapes'])-1
            best_mape = res['val_mapes'][best_epoch]
            print(f"  Best validation MAPE: {best_mape:.2f}%")
    
    # 5) Plot comparison
    plot_comparison(results, experiment_dir)
    
    # 6) Save comparison results
    comparison_data = {}
    for model_type, res in results.items():
        for metric in ['train_losses', 'val_losses', 'train_mapes', 'val_mapes', 'train_mean_smapes', 'val_mean_smapes', 'train_median_smapes', 'val_median_smapes', 'phasor_values', 'train_phasor_h1', 'val_phasor_h1', 'train_phasor_h2', 'val_phasor_h2']:
            comparison_data[f"{model_type}_{metric}"] = res[metric]
    
    comparison_path = os.path.join(experiment_dir, "comparison_results.npz")
    np.savez(comparison_path, **comparison_data)
    print(f"\nSaved comparison results to: {comparison_path}")
    
    # 7) Display clear recommendation for the best lambda value
    best_model = min(results.keys(), key=lambda k: results[k]['best_val_loss'])
    best_loss = results[best_model]['best_val_loss']
    best_lambda = best_model.split('_')[-1]  # Extract lambda value from model name
    
    print("\n"+"-"*50)
    print(f"RECOMMENDATION: The best lambda_phasor value is {best_lambda}")
    print(f"Best validation loss: {best_loss:.6f}")
    print(f"Total data loading time: {data_load_time:.2f} seconds (PARALLEL)")
    print("-"*50)
    
    print("\nParallel training and comparison complete.") 