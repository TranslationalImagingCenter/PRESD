"""Train PRESD on input / ground-truth pairs made by prepare_data.py.

Each folder passed with --data_folders is one prepared sample folder:
    <folder>/input/<name>.lsm   (R, C_in, H, W)  reduced spectral input
    <folder>/gt/<name>.lsm      (32, H, W)       mean of all 10 raw repeats
Input and GT files are paired by filename. C_in and R are read from the input
files (all files must agree). The excitation wavelength of each file is taken
from "Block N" in its filename (WAVELENGTH_BLOCK_N in global_val.txt).
"""
import argparse
import json
import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.optim as optim

from config import load_global_params, set_channel_num
from NN import PRESD, combined_spectral_loss, phasor_loss
from helper import (
    normalize_data, invert_scale, file_to_block_number, file_to_wavelength,
    compute_mape, compute_smape, set_seeds, initialize_directories, get_device,
)
from plot import plot_comparison

global_params = load_global_params()


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
def process_single_file(args):
    """Load one input/GT file pair and randomly sample pixels from it.

    Runs in a worker process, so everything it needs is passed in `args`:
        (input_folder, filename, gt_folder, file_index, samples_per_file, seed, norm)

    Returns:
        (x, y, w) with shapes (S, 1, C_in, R), (S, 32), (S,), or None on failure.
    """
    folder, f, gt_folder, file_index, samples_per_file, seed, norm = args

    try:
        f_path = os.path.join(folder, f)
        data = tifffile.TiffFile(f_path).asarray()   # (R, C_in, H, W)
        if data.ndim != 4:
            print(f"    [Process] Expected input shape (R, C_in, H, W), got {data.shape}; skipping {f}.")
            return None
        R, C, H, W = data.shape
        M = H * W
        wavelength = file_to_wavelength(f)
        x_in = data.reshape(R, C, M).transpose(1, 0, 2)    # (C_in, R, M)

        gt_path = os.path.join(gt_folder, f)
        if not os.path.isfile(gt_path):
            print(f"    [Process] GT file not found, skipping {f}: {gt_path}")
            return None
        gt = tifffile.TiffFile(gt_path).asarray()
        if gt.shape != (global_params["OUTPUT_CHANNELS"], H, W):
            print(f"    [Process] GT shape {gt.shape} does not match input {data.shape}; skipping {f}.")
            return None
        y_out = gt.reshape(global_params["OUTPUT_CHANNELS"], M)   # (32, M)

        # Per-file seed so sampling is reproducible regardless of process scheduling.
        rng = np.random.RandomState(seed + file_index)
        if M > samples_per_file:
            idx = rng.choice(M, samples_per_file, replace=False)
            x_in = x_in[:, :, idx]
            y_out = y_out[:, idx]

        x_in, y_out = normalize_data(x_in, y_out, norm)

        x_data = np.transpose(x_in, (2, 0, 1))[:, np.newaxis]    # (S, 1, C_in, R)
        y_data = y_out.T                                          # (S, 32)
        w_data = np.full((x_data.shape[0],), wavelength, dtype=np.float32)

        print(f"    [Process] Processed {f}: shape={data.shape}, wavelength={wavelength:.0f} nm, samples={x_data.shape[0]}")
        return x_data, y_data, w_data

    except Exception as e:
        print(f"    [Process] Error processing {f}: {e}")
        return None


def load_data(data_folders, selected_block=None, num_processes=None):
    """Load all input/GT pairs in parallel and split the sampled pixels into train/val sets.

    Args:
        data_folders:   prepared sample folders, each with input/ and gt/ sub-folders
        selected_block: if set, only files whose path contains "Block <selected_block>"
        num_processes:  worker processes (default: 75% of CPUs)
    """
    valid_files = []
    for data_folder in data_folders:
        folder = os.path.join(data_folder, "input")
        gt_folder = os.path.join(data_folder, "gt")
        if not os.path.isdir(folder):
            print(f"Input folder not found: {folder}")
            continue
        files = sorted(f for f in os.listdir(folder) if f.endswith(global_params["FILE_EXTENSION"]))
        for f in files:
            if selected_block is not None and file_to_block_number(f) != selected_block:
                continue
            valid_files.append((folder, f, gt_folder))

    if not valid_files:
        raise ValueError("No valid files found.")

    print(f"\nFound {len(valid_files)} valid files.")
    samples_per_file = global_params["MAX_SAMPLES"] // len(valid_files)
    print(f"Taking up to {samples_per_file} samples per file.")

    if num_processes is None:
        num_processes = max(1, int(cpu_count() * 0.75))
    num_processes = max(1, min(num_processes, len(valid_files)))
    print(f"Using {num_processes} parallel processes (out of {cpu_count()} available CPUs)")

    seed = global_params["SEED"]
    process_args = [
        (folder, f, gt_folder, i + 1, samples_per_file, seed, global_params["NORM"])
        for i, (folder, f, gt_folder) in enumerate(valid_files)
    ]

    start_time = time.time()
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_single_file, process_args)
    print(f"Parallel processing completed in {time.time() - start_time:.2f} seconds")

    results = [r for r in results if r is not None]
    print(f"Successfully processed {len(results)} out of {len(valid_files)} files")
    if not results:
        raise ValueError("No valid data found after parallel processing.")

    input_shapes = {r[0].shape[2:] for r in results}
    if len(input_shapes) > 1:
        raise ValueError(f"Input files disagree on (C_in, R): {sorted(input_shapes)}. "
                         "Prepare all training folders with the same --channels and --repeats.")

    x_data = np.concatenate([r[0] for r in results], axis=0)
    y_data = np.concatenate([r[1] for r in results], axis=0)
    w_data = np.concatenate([r[2] for r in results], axis=0)
    total_samples = x_data.shape[0]
    print(f"\nTotal collected samples: {total_samples}")

    # Re-seeding the global RNG here also fixes the batch order used later in training.
    np.random.seed(seed)
    inds = np.random.permutation(total_samples)
    n_train = int(global_params["training_ratio"] * total_samples)
    train_idx, val_idx = inds[:n_train], inds[n_train:]
    x_train, x_val = x_data[train_idx], x_data[val_idx]
    y_train, y_val = y_data[train_idx], y_data[val_idx]
    w_train, w_val = w_data[train_idx], w_data[val_idx]

    print(f"\nTrain set => x:{x_train.shape}, y:{y_train.shape}, wave:{w_train.shape}")
    print(f"Val set   => x:{x_val.shape}, y:{y_val.shape}, wave:{w_val.shape}")
    return x_train, w_train, y_train, x_val, w_val, y_val

# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------
def _create_model_state(model, epoch, val_loss, metrics):
    return {
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
        'metrics': metrics,
    }


def _predict_batched(model, x_t, w_t, batch_size):
    return np.vstack([
        model(x_t[s:s + batch_size], w_t[s:s + batch_size]).cpu().numpy()
        for s in range(0, x_t.shape[0], batch_size)
    ])


def _compute_epoch_metrics(model, x_train_t, w_train_t, y_train_t, x_val_t, w_val_t, y_val_t, loss_fn, norm_type):
    """Evaluate the model on the full train and val sets.

    val_loss and phasor_val are on normalized data; MAPE, SMAPE and the
    phasor_h1/h2 metrics are on the original intensity scale.
    """
    model.eval()
    eval_bs = global_params["EVAL_BATCH_SIZE"]
    with torch.no_grad():
        train_pred = _predict_batched(model, x_train_t, w_train_t, eval_bs)
        train_true = y_train_t.cpu().numpy()

        val_preds = []
        n_val = x_val_t.shape[0]
        val_loss_accum = 0.0
        phasor_val_accum = 0.0
        for s in range(0, n_val, eval_bs):
            e = min(s + eval_bs, n_val)
            batch_x, batch_w, batch_y = x_val_t[s:e], w_val_t[s:e], y_val_t[s:e]
            batch_pred = model(batch_x, batch_w)
            val_preds.append(batch_pred.cpu().numpy())
            val_loss_accum += loss_fn(batch_pred, batch_y, batch_x).item() * (e - s)
            phasor_val_accum += phasor_loss(batch_pred, batch_y, harmonic=global_params["DEFAULT_HARMONIC"]).item() * (e - s)
        val_pred = np.vstack(val_preds)
        val_true = y_val_t.cpu().numpy()

    train_pred_orig, train_true_orig = invert_scale(train_pred, train_true, norm_type)
    val_pred_orig, val_true_orig = invert_scale(val_pred, val_true, norm_type)

    h1, h2 = global_params["PHASOR_HARMONICS"]

    def _phasor(pred, true, h):
        return phasor_loss(torch.tensor(pred, dtype=torch.float32), torch.tensor(true, dtype=torch.float32), harmonic=h).item()

    train_mean_smape, train_median_smape = compute_smape(train_true_orig, train_pred_orig)
    val_mean_smape, val_median_smape = compute_smape(val_true_orig, val_pred_orig)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'val_loss': val_loss_accum / n_val,
        'phasor_val': phasor_val_accum / n_val,
        'train_mape': compute_mape(train_true_orig, train_pred_orig),
        'val_mape': compute_mape(val_true_orig, val_pred_orig),
        'train_mean_smape': train_mean_smape,
        'val_mean_smape': val_mean_smape,
        'train_median_smape': train_median_smape,
        'val_median_smape': val_median_smape,
        'train_phasor_h1': _phasor(train_pred_orig, train_true_orig, h1),
        'val_phasor_h1': _phasor(val_pred_orig, val_true_orig, h1),
        'train_phasor_h2': _phasor(train_pred_orig, train_true_orig, h2),
        'val_phasor_h2': _phasor(val_pred_orig, val_true_orig, h2),
    }


def _print_gpu_memory(gpu_ids, indent="  ", peak=False):
    for gpu_id in gpu_ids:
        allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
        reserved = torch.cuda.memory_reserved(gpu_id) / 1e9
        line = f"{indent}GPU {gpu_id}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved"
        if peak:
            line += f", {torch.cuda.max_memory_allocated(gpu_id) / 1e9:.2f} GB peak"
        print(line)


def train_model(model, x_train, w_train, y_train, x_val, w_val, y_val,
                lambda_phasor=None, harmonic=None, epochs=None, batch_size=None):
    """Train with MSE + consistency + phasor loss; returns the best checkpoint and per-epoch metrics."""
    start_time = time.time()
    if lambda_phasor is None:
        lambda_phasor = global_params["LAMBDA_PHASOR"]
    if harmonic is None:
        harmonic = global_params["DEFAULT_HARMONIC"]
    if epochs is None:
        epochs = global_params["EPOCHS"]
    if batch_size is None:
        batch_size = global_params["BATCHSIZE"]
    norm_type = global_params["NORM"]

    gpu_ids = global_params["DEFAULT_GPUS"]
    device = get_device(gpu_ids[0])
    print(f"Training device: {device}")
    model.to(device)

    if torch.cuda.is_available():
        print(f"Using GPUs {' & '.join(map(str, gpu_ids))} for training")
        for gpu_id in gpu_ids:
            props = torch.cuda.get_device_properties(gpu_id)
            print(f"  GPU {gpu_id}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print("GPU memory after model setup:")
        _print_gpu_memory(gpu_ids)

    initial_lr = global_params["INITIAL_LR"]
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=global_params["WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=global_params["LR_FACTOR"],
        patience=global_params["LR_PATIENCE"], min_lr=global_params["MIN_LR"],
    )

    # The whole dataset is moved to the device up front.
    x_train_t = torch.tensor(x_train, dtype=torch.float32).to(device)
    w_train_t = torch.tensor(w_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).to(device)
    w_val_t = torch.tensor(w_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    lambda_consistency = global_params["LAMBDA_CONSISTENCY"]
    print(f"Training with MSE + consistency + phasor loss "
          f"(lambda_consistency={lambda_consistency}, lambda_phasor={lambda_phasor}, harmonic={harmonic})")

    def loss_fn(pred, target, input_x):
        return combined_spectral_loss(pred, target, input_x, lambda_consistency, lambda_phasor, harmonic)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stop_patience = global_params["EARLY_STOP_PATIENCE"]
    warmup_epochs = global_params["WARMUP_EPOCHS"]
    grad_clip_max_norm = global_params["GRAD_CLIP_MAX_NORM"]

    metrics = {
        'train_losses': [], 'val_losses': [], 'train_mapes': [], 'val_mapes': [],
        'train_mean_smapes': [], 'val_mean_smapes': [], 'train_median_smapes': [], 'val_median_smapes': [],
        'phasor_values': [], 'train_phasor_h1': [], 'val_phasor_h1': [],
        'train_phasor_h2': [], 'val_phasor_h2': [],
    }
    # epoch-metric key → history key in `metrics`
    history_keys = {
        'val_loss': 'val_losses', 'phasor_val': 'phasor_values',
        'train_mape': 'train_mapes', 'val_mape': 'val_mapes',
        'train_mean_smape': 'train_mean_smapes', 'val_mean_smape': 'val_mean_smapes',
        'train_median_smape': 'train_median_smapes', 'val_median_smape': 'val_median_smapes',
        'train_phasor_h1': 'train_phasor_h1', 'val_phasor_h1': 'val_phasor_h1',
        'train_phasor_h2': 'train_phasor_h2', 'val_phasor_h2': 'val_phasor_h2',
    }

    n_samples = x_train_t.shape[0]
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        perm = np.random.permutation(n_samples)
        batch_losses = []

        # Linear warmup; after warmup the LR is left to the scheduler.
        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr * (epoch + 1) / warmup_epochs

        for start_idx in range(0, n_samples, batch_size):
            idx = perm[start_idx:start_idx + batch_size]
            batch_x, batch_w, batch_y = x_train_t[idx], w_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            pred = model(batch_x, batch_w)
            loss = loss_fn(pred, batch_y, batch_x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            optimizer.step()
            batch_losses.append(loss.item())

        epoch_train_loss = float(np.mean(batch_losses))
        metrics['train_losses'].append(epoch_train_loss)
        epoch_metrics = _compute_epoch_metrics(model, x_train_t, w_train_t, y_train_t,
                                               x_val_t, w_val_t, y_val_t, loss_fn, norm_type)
        for key, hist_key in history_keys.items():
            metrics[hist_key].append(epoch_metrics[key])

        scheduler.step(epoch_metrics['val_loss'])

        if epoch_metrics['val_loss'] < best_val_loss:
            best_val_loss = epoch_metrics['val_loss']
            best_model_state = _create_model_state(model, epoch, best_val_loss, metrics)
            patience_counter = 0
        else:
            patience_counter += 1

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {epoch_train_loss:.6f}, "
              f"Val Loss: {epoch_metrics['val_loss']:.6f}, "
              f"Train MAPE: {epoch_metrics['train_mape']:.2f}%, "
              f"Val MAPE: {epoch_metrics['val_mape']:.2f}%, "
              f"Val Mean_SMAPE: {epoch_metrics['val_mean_smape']:.2f}%, "
              f"Val Median_SMAPE: {epoch_metrics['val_median_smape']:.2f}%, "
              f"Time: {time.time() - epoch_start_time:.2f}s")
        print(f"            Phasor(h={global_params['PHASOR_HARMONICS'][0]}): "
              f"Train={epoch_metrics['train_phasor_h1']:.6f}, Val={epoch_metrics['val_phasor_h1']:.6f}")
        print(f"            Phasor(h={global_params['PHASOR_HARMONICS'][1]}): "
              f"Train={epoch_metrics['train_phasor_h2']:.6f}, Val={epoch_metrics['val_phasor_h2']:.6f}, LR={lr:.6f}")

        if torch.cuda.is_available() and (epoch + 1) % 5 == 0:
            print(f"            GPU memory (epoch {epoch+1}):")
            _print_gpu_memory(gpu_ids, indent="              ")

        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    if torch.cuda.is_available():
        print("Final GPU memory summary:")
        _print_gpu_memory(gpu_ids, peak=True)

    # Only happens if the validation loss was never finite (e.g. NaN from the first epoch).
    if best_model_state is None:
        best_model_state = _create_model_state(model, epoch, metrics['val_losses'][-1], metrics)

    return {
        'model_state': best_model_state,
        'training_time': training_time,
        'best_epoch': best_model_state['epoch'],
        'best_val_loss': best_model_state['val_loss'],
        **metrics,
    }


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train PRESD on input / GT pairs made by prepare_data.py")
    parser.add_argument('--data_folders', type=str, nargs='+', required=True,
                        help='Prepared sample folders, each containing input/ and gt/ (output of prepare_data.py)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Root directory for training outputs (overrides PLOTS_DIR in global_val.txt)')
    parser.add_argument('--harmonic', type=int, default=global_params["DEFAULT_HARMONIC"],
                        help=f'Phasor harmonic used in the loss (default: {global_params["DEFAULT_HARMONIC"]})')
    parser.add_argument('--block', type=str, default='all',
                        help='Only train on files whose name contains "Block N" for this N, or "all" (default: all)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Save as <model_name>.pt / <model_name>_metrics.json (default: NN_<C>ch_<R>repeats)')
    parser.add_argument('--model_dir', type=str, default=global_params["USER_MODELS_DIR"],
                        help=f'Root folder for trained weights (default: {global_params["USER_MODELS_DIR"]}). '
                             'The pre-trained models/ folder is never written to.')
    parser.add_argument('--processes', type=int, default=None,
                        help='Number of data-loading processes (default: 75%% of available CPUs)')
    args = parser.parse_args()

    set_seeds(global_params["SEED"])

    print("\nGPU information:")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    elif torch.backends.mps.is_available():
        print("CUDA not available - using Apple GPU (MPS)")
    else:
        print("CUDA and Apple GPU (MPS) not available - using CPU only")

    selected_block = None
    if args.block.lower() != 'all':
        try:
            selected_block = int(args.block)
            print(f"Training on block {selected_block} only")
        except ValueError:
            print(f"Invalid block value: {args.block}. Using all blocks.")
    else:
        print("Training on all blocks")

    print("\n" + "=" * 60 + "\nLOADING DATA\n" + "=" * 60)
    data_start_time = time.time()
    x_train, w_train, y_train, x_val, w_val, y_val = load_data(args.data_folders, selected_block, args.processes)
    data_load_time = time.time() - data_start_time
    print(f"Data loading completed in {data_load_time:.2f} seconds")

    # C_in and R come from the prepared input files.
    channel_num, num_repeats = x_train.shape[2], x_train.shape[3]
    set_channel_num(channel_num)

    lambda_phasor = global_params["LAMBDA_PHASOR"]
    print("\nTraining settings:")
    print(f"  Input channels (C_in): {channel_num}")
    print(f"  Repeats (R):           {num_repeats}")
    print(f"  Harmonic:              {args.harmonic}")
    print(f"  Lambda phasor:         {lambda_phasor}")
    print(f"  Data folders:          {args.data_folders}")

    # Outputs go to <output_dir>/train/<C>_ch/<sample folder names>_<R>repeats/
    sample_names = "_".join(os.path.basename(os.path.normpath(f)) for f in args.data_folders)
    if args.output_dir is not None:
        global_params["PLOTS_DIR"] = args.output_dir
    experiment_dir = os.path.join(initialize_directories(), "train", f"{channel_num}_ch",
                                  f"{sample_names}_{num_repeats}repeats")
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Training results will be saved to: {os.path.abspath(experiment_dir)}")

    block_info = f"_block{selected_block}" if selected_block is not None else ""
    run_name = f"Phasor_lambda_{lambda_phasor}{block_info}"
    print(f"\n===== Training model: {run_name} =====")

    model = PRESD(replicate_count=num_repeats)
    results = train_model(
        model, x_train, w_train, y_train, x_val, w_val, y_val,
        lambda_phasor=lambda_phasor, harmonic=args.harmonic,
    )

    # Same naming as the pre-trained models in models/, so test.py usage is identical.
    model_name = args.model_name or f"NN_{channel_num}ch_{num_repeats}repeats{block_info}"
    model_dir = os.path.join(args.model_dir, f"{channel_num}_ch", f"{sample_names}_{num_repeats}repeats")
    pretrained_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
    if os.path.commonpath([os.path.realpath(model_dir), pretrained_dir]) == pretrained_dir:
        raise SystemExit(f"Refusing to save into the pre-trained models folder ({pretrained_dir}); "
                         "choose another --model_dir.")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.pt")
    torch.save(results['model_state'], model_path)
    print(f"Saved model to: {os.path.abspath(model_path)}")

    metrics_path = os.path.join(experiment_dir, f"{model_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({k: v for k, v in results.items() if k != 'model_state'}, f, indent=global_params["JSON_INDENT"])
    print(f"Saved metrics to: {os.path.abspath(metrics_path)}")

    plot_comparison({run_name: results}, experiment_dir)

    history_keys = ['train_losses', 'val_losses', 'train_mapes', 'val_mapes',
                    'train_mean_smapes', 'val_mean_smapes', 'train_median_smapes', 'val_median_smapes',
                    'phasor_values', 'train_phasor_h1', 'val_phasor_h1', 'train_phasor_h2', 'val_phasor_h2']
    comparison_path = os.path.join(experiment_dir, "comparison_results.npz")
    np.savez(comparison_path, **{f"{run_name}_{k}": results[k] for k in history_keys})
    print(f"Saved per-epoch metrics to: {comparison_path}")

    best_epoch = results['best_epoch']
    print("\n" + "-" * 50)
    print(f"Best epoch: {best_epoch + 1}")
    print(f"  Validation loss: {results['best_val_loss']:.6f}")
    print(f"  Validation MAPE: {results['val_mapes'][best_epoch]:.2f}%")
    print(f"  Training time:   {results['training_time']:.2f} seconds")
    print(f"  Data loading:    {data_load_time:.2f} seconds")
    print("-" * 50)


if __name__ == "__main__":
    main()
