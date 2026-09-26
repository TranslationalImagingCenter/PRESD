# PRESD

A deep learning pipeline for **spectral super-resolution** of fluorescence microscopy images. Given a low-channel (4, 8, or 16 spectral channels) `.lsm` acquisition, the model reconstructs the full **32-channel** emission spectrum per pixel.

---

## Overview

Fluorescence spectral microscopy typically requires many spectral channels to resolve overlapping fluorophore signatures. This project trains a neural network (FiLM-conditioned attention network) to upscale low-channel spectral images to 32 channels, enabling faster acquisitions without sacrificing spectral resolution.

**Key features:**
- Accepts native 4, 8, or 16-channel `.lsm` input files directly
- Supports multiple acquisition repeats as input to improve reconstruction quality
- Excitation wavelength passed explicitly via `--wavelength` for accurate FiLM conditioning
- Outputs full 32-channel prediction as `.lsm`, per-pixel metric maps, and spectral plots
- Metrics (MAPE, SMAPE, RMSE, phasor distance) computed automatically when a ground-truth file is provided

---

## Repository Structure

```
.
├── train.py                # Training script
├── test.py                 # Inference / evaluation script
├── NN.py                   # Model architecture (FilmEarlyNet)
├── config.py               # Parameter loader (reads global_val.txt)
├── helper.py               # Utilities: normalization, metrics, wavelength parsing
├── plot.py                 # Training visualisation helpers
├── global_val.txt          # All hyperparameters and paths (edit here)
├── requirements.txt        # Python dependencies
└── models/                 # Pre-trained model weights
    ├── NN_4ch_1repeats.pt … NN_4ch_10repeats.pt    # 4-channel models  (1–10 repeats)
    ├── NN_8ch_1repeats.pt … NN_8ch_10repeats.pt    # 8-channel models  (1–10 repeats)
    └── NN_16ch_1repeats.pt … NN_16ch_10repeats.pt  # 16-channel models (1–10 repeats)
```

Model files are named `NN_<channels>ch_<repeats>repeats.pt` — pick the one that matches the `--channels` and `--repeats` you intend to run.

---

## Requirements

**Python 3.11.12** (tested with the `spectra` conda environment)

```bash
pip install -r requirements.txt
```

| Package | Version |
|---|---|
| numpy | 1.26.4 |
| torch | 2.6.0 |
| tifffile | 2025.1.10 |
| matplotlib | 3.10.0 |
| scipy | 1.8.0 |
| tqdm | ≥ 4.66.0 |

A CUDA-capable GPU is strongly recommended for training. Inference runs on CPU as well.

---

## Downloading the Datasets

*(Dataset download links and instructions will be added here.)*

<!-- TODO: Add dataset download links, size, and extraction instructions. -->

Once downloaded, extract the data so that it sits **next to** (not inside) this repository, following the layout in the next section.

---

## Directory Layout & Where to Run

This repository holds only the **code and pre-trained models**. Datasets and outputs are kept in sibling folders one level *above* the repo, because the default paths in `global_val.txt` are relative (`../dataset`, `../results`, `../saved_inputs_human`).

The recommended layout is:

```
project-root/
├── PRESD/                    # ← this repository (contains train.py, test.py, models/, …)
├── dataset/                  # ← the downloaded datasets go here
│   ├── train_sample1/  train_sample2/  …   # training samples (folders of .lsm files)
│   └── test_sample1/   test_sample2/   …    # testing samples (folders of .lsm files)
├── results/                  # ← created automatically for training/inference outputs
└── saved_inputs_human/       # ← created automatically for cached processed data
```

All folder and file names shown here (`train_sample1`, `test_sample1`, `sample_01.lsm`, …) are just **examples**. The `train_sample*` folders hold the training samples and the `test_sample*` folders hold the testing samples — in both cases the data is stored as `.lsm` files. Use whatever names your downloaded dataset uses and point the scripts at them with the path arguments.

**Where to run commands:** always run `python train.py …` / `python test.py …` from **inside the repository directory** (the folder that contains `train.py`). All example paths below use `../` to reach the sibling `dataset/` and `results/` folders from there.

```bash
cd PRESD          # the directory containing train.py
python test.py …  # paths like ../dataset/... and ../results/... resolve correctly
```

You are not locked into these names — every script takes explicit path arguments (`--input_folders`, `--input_file`, `--gt_file`, `--output_dir`), so you can point them anywhere. The layout above simply matches the built-in defaults so the examples work out of the box.

---

## Data Format

### Input files
`.lsm` files with **N spectral channels** and **R acquisition repeats**:
```
shape = (num_repeats, num_channels, H, W)
```
- `num_channels` must be 4, 8, or 16 (must divide 32 evenly)
- `num_repeats` ≥ the number of repeats you pass via `--repeats`

### Ground-truth files
`.lsm` files with **32 spectral channels**:
```
shape = (32, H, W)
```
GT files must share the **same filename** as their paired input files when used for training.

---

## Usage

### Training

Train a model using paired input and ground-truth `.lsm` files. The input folder and GT folder must contain files with **matching filenames** (e.g. `Image_11_Block_1.lsm` in both):

```
dataset/train_sample1/
├── input/
│   ├── sample_01.lsm    # shape (num_repeats, num_channels, H, W)
│   ├── sample_02.lsm
│   └── ...
└── gt/
    ├── sample_01.lsm    # shape (32, H, W) — same filename as input
    ├── sample_02.lsm
    └── ...
```

Run all commands below from the repository directory (the folder that contains `train.py`); the `../` paths reach the sibling `dataset/` and `results/` folders.

**Example 1 — train with 5 repeats, 4 channels (the tested configuration):**
```bash
python train.py \
    --repeats 0 1 2 3 4 \
    --channels 4 \
    --input_folders ../dataset/train_sample1/input \
    --gt_folders    ../dataset/train_sample1/gt \
    --output_dir    ../results
```

**Example 2 — train with 1 repeat, 4 channels:**
```bash
python train.py \
    --repeats 0 \
    --channels 4 \
    --input_folders ../dataset/train_sample1/input \
    --gt_folders    ../dataset/train_sample1/gt \
    --output_dir    ../results
```

**Example 3 — train on multiple datasets simultaneously:**
```bash
python train.py \
    --repeats 0 1 2 \
    --channels 4 \
    --input_folders ../dataset/train_sample1/input ../dataset/train_sample2/input \
    --gt_folders    ../dataset/train_sample1/gt    ../dataset/train_sample2/gt \
    --output_dir    ../results
```

**Training arguments:**

| Argument | Required | Description |
|---|---|---|
| `--repeats` | Yes | Repeat indices to use, e.g. `0 1 2 3 4` |
| `--channels` | Yes | Number of input spectral channels (4, 8, or 16) |
| `--input_folders` | Yes | Directories containing the input `.lsm` files |
| `--gt_folders` | Yes | Paired directories containing the GT `.lsm` files |
| `--output_dir` | No | Root output directory (overrides `PLOTS_DIR` in `global_val.txt`) |

Trained models are saved to `<output_dir>/train/<C>_ch/<experiment>/NN.pt`.

---

### Inference (test)

Run all commands below from the repository directory (the folder that contains `test.py`). The `--model_path` points at the bundled `models/` folder inside the repo, while `--input_file` and `--output_dir` use `../` to reach the sibling `dataset/` and `results/` folders.

**With ground truth (computes metrics and comparison plots):**
```bash
python test.py \
    --repeats 4 \
    --channels 4 \
    --wavelength 561 \
    --model_path models/NN_4ch_4repeats.pt \
    --input_file ../dataset/test_sample1/input.lsm \
    --output_dir ../results/test_sample1 \
    --gt_file    ../dataset/test_sample1/ground_truth.lsm
```

**Without ground truth (prediction only):**
```bash
python test.py \
    --repeats 4 \
    --channels 4 \
    --wavelength 561 \
    --model_path models/NN_4ch_4repeats.pt \
    --input_file ../dataset/test_sample1/input.lsm \
    --output_dir ../results/test_sample1
```

> **Tip:** match the model to your inputs — use `models/NN_<channels>ch_<repeats>repeats.pt`. For example, `--channels 8 --repeats 3` pairs with `models/NN_8ch_3repeats.pt`.

**Inference arguments:**

| Argument | Required | Description |
|---|---|---|
| `--repeats` | Yes | Number of repeats to use (e.g. `5` → uses indices 0–4) |
| `--channels` | Yes | Number of input spectral channels (4, 8, or 16) |
| `--wavelength` | Yes | Excitation wavelength in nm (405, 458, 488, 514, 561, 591, or 633) |
| `--model_path` | Yes | Path to the `.pt` model checkpoint |
| `--input_file` | Yes | Path to the input `.lsm` file |
| `--output_dir` | Yes | Directory where outputs are saved |
| `--gt_file` | No | Paired 32-channel GT `.lsm` file for metric computation |

---

## Outputs

All outputs are saved under `--output_dir`:

| File / Folder | Condition | Description |
|---|---|---|
| `pred32.lsm` | Always | Predicted 32-channel image, shape `(32, H, W)` |
| `metrics.json` | GT provided | Overall MAPE, SMAPE, RMSE, Phasor h1/h2 |
| `channels/channel_XX.png` | Always | Per-channel spatial images. With GT: 5 panels (GT \| Pred \| SMAPE \| RMSE \| Phasor h2). Without GT: Pred only. |
| `spectra/spectrum_XX.png` | Always | 50 random pixel spectra showing Pred (red), GT (blue, if available), and input as a step/box plot (green) |

---

## Configuration

All parameters are set in `global_val.txt`. Key settings:

| Parameter | Description |
|---|---|
| `channel_num` | Default number of input channels |
| `EPOCHS` | Number of training epochs |
| `BATCHSIZE` | Training batch size |
| `INITIAL_LR` | Initial learning rate |
| `LAMBDA_PHASOR` | Weight for the phasor loss term |
| `NORM` | Normalization type (`log` or `linear`) |
| `PLOTS_DIR` | Default root output directory |
| `WAVELENGTH_BLOCK_1…7` | Excitation wavelengths per block (405–633 nm) |

---

## Model Architecture

`FilmEarlyNet` (`NN.py`) — a hybrid 2D→1D convolutional network with:
- **FiLM conditioning** on excitation wavelength
- **Multi-head self-attention** over the spectral-repeat plane
- **Phasor-aware loss** penalising both spectral shape and phasor (G, S) coordinates

Input shape: `(batch, 1, num_channels, num_repeats)`  
Output shape: `(batch, 32)`

---

## License

Released under the MIT License — see the [`LICENSE`](LICENSE) file for details.
