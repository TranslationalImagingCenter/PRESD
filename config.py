"""Loads the parameters in global_val.txt into a dictionary."""
import os

import numpy as np

# Keys whose value is a space-separated list of numbers.
_NUMERIC_LIST_KEYS = {"DEFAULT_GPUS", "PHASOR_HARMONICS", "FIGURE_SIZE_LARGE"}

# Supported numbers of input channels (must divide OUTPUT_CHANNELS).
VALID_CHANNEL_NUMS = [4, 8, 16]

# Every module shares one parameter dict, so the channel count that
# train.py / test.py read from the data (set_channel_num) is seen everywhere.
_PARAMS = None


def load_global_params():
    """Return the shared parameter dict, reading global_val.txt on first call."""
    global _PARAMS
    if _PARAMS is None:
        _PARAMS = _read_params_file()
    return _PARAMS


def set_channel_num(channel_num):
    """Set the number of input channels (C_in) and the values derived from it."""
    params = load_global_params()
    if channel_num not in VALID_CHANNEL_NUMS:
        raise ValueError(f"channel_num must be one of {VALID_CHANNEL_NUMS}")
    params["channel_num"] = channel_num
    params["GROUP_SIZE"] = params["OUTPUT_CHANNELS"] // channel_num


def _read_params_file():
    global_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "global_val.txt")
    if not os.path.exists(global_file):
        raise FileNotFoundError(f"Global parameters file not found: {global_file}")

    params = {}
    with open(global_file, "r") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            key, value_str = parts[0], parts[1].strip()
            params[key] = _parse_value(key, value_str)

    # Excitation wavelength (nm) for each acquisition block: {block_number: nm}.
    params["WAVELENGTH_MAP"] = {
        int(k.rsplit("_", 1)[1]): v for k, v in params.items() if k.startswith("WAVELENGTH_BLOCK_")
    }
    params["CHAN_WAVELENGTHS_ARRAY"] = np.array(params["CHAN_WAVELENGTHS"], dtype=np.float32)
    params["VALID_CHANNEL_NUMS"] = VALID_CHANNEL_NUMS
    return params


def _parse_value(key, value_str):
    if key == "CHAN_WAVELENGTHS":
        return [int(x) for x in value_str.split(",")]
    if key in _NUMERIC_LIST_KEYS:
        return [_to_number(x) for x in value_str.split()]
    if value_str.lower() in ("true", "false"):
        return value_str.lower() == "true"
    try:
        return _to_number(value_str)
    except ValueError:
        return value_str


def _to_number(s):
    """Parse s as an int if possible, otherwise as a float."""
    try:
        return int(s)
    except ValueError:
        return float(s)
