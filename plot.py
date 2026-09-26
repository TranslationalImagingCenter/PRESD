import os
import numpy as np
import matplotlib.pyplot as plt
import json
from config import load_global_params

# Load global parameters
global_params = load_global_params()

def _setup_subplot(title, xlabel="Epoch", ylabel="", log_scale=False):
    """Helper to setup common subplot styling."""
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if log_scale:
        plt.yscale('log')
    plt.grid(True, alpha=global_params["GRID_ALPHA"])
    plt.legend()

def plot_comparison(results, plots_dir):
    """Plot comparison of training metrics for different lambda_phasor values."""
    # Define plot configurations
    plots_config = [
        ('val_losses', "Validation Loss", "Loss", True),
        ('train_losses', "Training Loss", "Loss", True),
        ('phasor_values', "Phasor Loss Component", "Phasor Loss", False),
        ('val_mapes', "Validation MAPE", "MAPE (%)", False),
        ('train_mapes', "Training MAPE", "MAPE (%)", False)
    ]
    
    plt.figure(figsize=global_params["FIGURE_SIZE_LARGE"])
    
    # Create subplots for metrics
    for i, (metric_key, title, ylabel, log_scale) in enumerate(plots_config, 1):
        plt.subplot(2, 3, i)
        for model_type, res in results.items():
            plt.plot(res[metric_key], label=model_type)
        _setup_subplot(title, ylabel=ylabel, log_scale=log_scale)
    
    # Training time comparison
    plt.subplot(2, 3, 6)
    model_types = list(results.keys())
    training_times = [results[m]['training_time'] for m in model_types]
    bars = plt.bar(model_types, training_times)
    _setup_subplot("Training Time", xlabel="", ylabel="Time (seconds)")
    plt.xticks(rotation=global_params["ROTATION_ANGLE"])
    # Add value labels on bars
    for bar, val in zip(bars, training_times):
        plt.text(bar.get_x() + bar.get_width()/2., 
                bar.get_height() + bar.get_height()*0.01, 
                f"{val:.1f}s", 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "loss_comparison.png"), dpi=global_params["DPI"])
    plt.close()