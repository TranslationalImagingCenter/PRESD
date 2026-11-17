"""Create input / ground-truth .lsm pairs from raw 32-channel PRESD acquisitions.

Each raw file holds 10 repeated 32-channel acquisitions, shape (10, 32, H, W).
For every raw file this writes, under <output_dir>/<sample folder name>/:

    input/<filename>  (R, C_in, H, W)  the selected repeats, each reduced from
                                       32 to C_in channels by averaging groups
                                       of 32 / C_in adjacent channels
    gt/<filename>     (32, H, W)       the mean of all 10 repeats

Filenames are kept, so the excitation wavelength can still be read from
"Block N" in the name. Run it separately for the training and the test samples.

Example:
    python prepare_data.py --raw_folders ../dataset/human/samp5 ../dataset/human/samp10 \\
        --channels 4 --repeats 1 2 3 4 5 --output_dir ../dataset/prepared/train_4ch_rep_1-5
"""
import argparse
import json
import os

import numpy as np
import tifffile

from config import load_global_params

global_params = load_global_params()

NUM_RAW_REPEATS = 10      # repeats per raw acquisition; the GT is their mean
CHANNEL_CHOICES = [4, 8, 16]


def load_raw(path):
    """Read a raw acquisition as (repeats, 32, H, W), dropping any singleton axes."""
    data = np.squeeze(tifffile.TiffFile(path).asarray())
    n_out = global_params["OUTPUT_CHANNELS"]
    if data.ndim != 4 or data.shape[1] != n_out or data.shape[0] < NUM_RAW_REPEATS:
        raise ValueError(f"expected shape ({NUM_RAW_REPEATS}, {n_out}, H, W), got {data.shape}")
    return data[:NUM_RAW_REPEATS].astype(np.float32)


def make_input(raw, channels, repeats):
    """Bin each selected repeat from 32 to `channels` channels → (len(repeats), channels, H, W).

    `repeats` are 1-based repeat numbers.
    """
    R, n_out, H, W = raw.shape
    group_size = n_out // channels
    selected = raw[[r - 1 for r in repeats]]                                  # (R_sel, 32, H, W)
    return selected.reshape(len(repeats), channels, group_size, H, W).mean(axis=2)


def make_gt(raw):
    """Mean of all repeats → (32, H, W)."""
    return raw.mean(axis=0)


def main():
    parser = argparse.ArgumentParser(description="Create input / GT .lsm pairs from raw 32-channel acquisitions")
    parser.add_argument("--raw_folders", nargs="+", required=True,
                        help="Folders of raw .lsm files, shape (10, 32, H, W), e.g. ../dataset/human/samp5")
    parser.add_argument("--channels", type=int, required=True, choices=CHANNEL_CHOICES,
                        help="Number of reduced input channels")
    parser.add_argument("--repeats", type=int, nargs="+", required=True,
                        help=f"Repeats to put in the input, any of 1-{NUM_RAW_REPEATS} (e.g. 1 2 3)")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write <sample>/input/ and <sample>/gt/")
    args = parser.parse_args()

    repeats = sorted(set(args.repeats))
    if repeats[0] < 1 or repeats[-1] > NUM_RAW_REPEATS:
        parser.error(f"--repeats must be between 1 and {NUM_RAW_REPEATS}")

    sample_names = [os.path.basename(os.path.normpath(f)) for f in args.raw_folders]
    if len(set(sample_names)) != len(sample_names):
        parser.error("--raw_folders must have different folder names (they name the output sub-folders)")

    ext = global_params["FILE_EXTENSION"]
    n_written = 0
    for folder, sample in zip(args.raw_folders, sample_names):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Raw folder not found: {folder}")
        files = sorted(f for f in os.listdir(folder) if f.endswith(ext))
        if not files:
            print(f"Warning: no {ext} files in {folder}")
            continue

        input_dir = os.path.join(args.output_dir, sample, "input")
        gt_dir = os.path.join(args.output_dir, sample, "gt")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

        print(f"{folder} → {os.path.join(args.output_dir, sample)}")
        for f in files:
            try:
                raw = load_raw(os.path.join(folder, f))
            except ValueError as e:
                print(f"  skipping {f}: {e}")
                continue
            x = make_input(raw, args.channels, repeats)
            y = make_gt(raw)
            tifffile.imwrite(os.path.join(input_dir, f), x)
            tifffile.imwrite(os.path.join(gt_dir, f), y)
            n_written += 1
            print(f"  {f}: input {x.shape}, gt {y.shape}")

    info = {
        "raw_folders": args.raw_folders,
        "channels": args.channels,
        "repeats": repeats,
        "gt": f"mean of all {NUM_RAW_REPEATS} repeats",
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "prepare_info.json"), "w") as jf:
        json.dump(info, jf, indent=global_params["JSON_INDENT"])

    print(f"\nWrote {n_written} input/GT pairs to {args.output_dir} "
          f"(channels={args.channels}, repeats={repeats})")


if __name__ == "__main__":
    main()
