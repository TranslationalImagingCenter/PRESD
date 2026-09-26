#!/usr/bin/env python3
"""
Configuration loader for global parameters.
Reads parameters from global_val.txt and converts them to appropriate types.
"""
import os
import numpy as np

def load_global_params():
    """Load global parameters from global_val.txt file."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    global_file = os.path.join(script_dir, "global_val.txt")
    
    if not os.path.exists(global_file):
        raise FileNotFoundError(f"Global parameters file not found: {global_file}")
    
    params = {}
    
    with open(global_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Split on first whitespace to separate key and value
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
                
            key = parts[0]
            value_str = parts[1]
            
            # Remove inline comments
            if '#' in value_str:
                value_str = value_str.split('#')[0].strip()
            
            # Parse different parameter types
            params[key] = _parse_parameter_value(key, value_str)
    
    # Post-processing for special parameters
    params = _post_process_parameters(params)
    
    return params

def _parse_parameter_value(key, value_str):
    """Parse parameter value based on key name and content."""
    
    # Handle empty values
    if not value_str or value_str.isspace():
        return ""
    
    # Boolean parameters
    if value_str.lower() in ['true', 'false']:
        return value_str.lower() == 'true'
    
    # Numeric list parameters (space-separated, convert to int/float)
    if key in ['POSITION_LIST', 'DEFAULT_GPUS', 'PHASOR_HARMONICS',
               'FIGURE_SIZE_LARGE', 'FIGURE_SIZE_MEDIUM', 'FIGURE_SIZE_SMALL', 'FIGURE_SIZE_HELPER',
               'VALID_CHANNEL_NUMS']:
        return [int(x) if x.replace('.', '').isdigit() else float(x) for x in value_str.split()]
    
    # Float array parameters (always convert to float)
    if key in ['CHAN_WAVELENGTHS_ARRAY']:
        return [float(x) for x in value_str.split()]
    
    # String list parameters (space-separated, keep as strings)
    # Note: TRAIN_FOLDERS is processed in post-processing to add base path
    if key in []:
        return value_str.split()
    
    # Comma-separated list (CHAN_WAVELENGTHS)
    if key == 'CHAN_WAVELENGTHS':
        return [int(x.strip()) for x in value_str.split(',')]
    
    # GPU mapping (special format: 4:0 8:1 16:2)
    if key == 'GPU_CHANNEL_MAPPING':
        mapping = {}
        for pair in value_str.split():
            if ':' in pair:
                ch, gpu = pair.split(':')
                mapping[int(ch)] = int(gpu)
        return mapping
    
    # Experiment positions (semicolon-separated lists)
    if key == 'EXPERIMENT_POSITIONS':
        position_lists = []
        for pos_list_str in value_str.split(';'):
            position_lists.append([int(x) for x in pos_list_str.split(',')])
        return position_lists
    
    # Directory paths (handle empty retina path)
    if 'PATH' in key or 'DIR' in key:
        return value_str.strip() if value_str.strip() else ""
    
    # Try to parse as number
    try:
        # Check for scientific notation
        if 'e-' in value_str.lower() or value_str.count('.') > 6:  # Many decimal places
            return float(value_str)
        
        # Try integer first
        if '.' not in value_str:
            return int(value_str)
        else:
            return float(value_str)
    except ValueError:
        pass
    
    # Return as string if all else fails
    return value_str.strip()

def _post_process_parameters(params):
    """Post-process parameters for special handling."""
    
    # Create wavelength mapping dictionary
    wavelength_map = {}
    for i in range(1, 8):  # Blocks 1-7
        key = f'WAVELENGTH_BLOCK_{i}'
        if key in params:
            wavelength_map[i] = params[key]
    params['WAVELENGTH_MAP'] = wavelength_map
    
    # Convert CHAN_WAVELENGTHS to numpy array if available
    if 'CHAN_WAVELENGTHS' in params:
        params['CHAN_WAVELENGTHS_ARRAY'] = np.array(params['CHAN_WAVELENGTHS'], dtype=np.float32)
    
    # Parse dataset folders from space-separated lists
    # (TRAIN_FOLDERS are all human samples; test folders that live under the
    #  feline dataset can be pointed at explicitly via the script arguments.)
    base_path = params.get('HUMAN_DATASET_BASE_PATH', '../dataset/human')
    train_folders_param = params.get('TRAIN_FOLDERS', '')
    test_folders_param = params.get('TEST_FOLDERS', '')
    
    # Create full paths for train and test folders
    train_folders = []
    if train_folders_param:
        # Handle both string (from config file) and list (from command line override)
        if isinstance(train_folders_param, str):
            for folder in train_folders_param.split():
                train_folders.append(os.path.join(base_path, folder))
        elif isinstance(train_folders_param, list):
            # Already processed by command line args, just use as-is
            train_folders = train_folders_param
    
    test_folders = []
    if test_folders_param:
        # Handle both string (from config file) and list (from command line override)
        if isinstance(test_folders_param, str):
            for folder in test_folders_param.split():
                test_folders.append(os.path.join(base_path, folder))
        elif isinstance(test_folders_param, list):
            # Already processed by command line args, just use as-is
            test_folders = test_folders_param
    
    params['TRAIN_FOLDERS'] = train_folders
    params['TEST_FOLDERS'] = test_folders
    params['ALL_FOLDERS'] = train_folders + test_folders
    
    # Simplified cache and plots directories
    params['CACHE_DIR'] = params.get('CACHE_DIR', '../saved_inputs_human')
    params['PLOTS_DIR'] = params.get('PLOTS_DIR', '../predict_plots/human')
    
    # Calculate derived parameters
    if 'channel_num' in params and 'GROUP_SIZE_DIVISOR' in params:
        params['GROUP_SIZE'] = params['GROUP_SIZE_DIVISOR'] // params['channel_num']
    
    # Valid channel numbers
    params['VALID_CHANNEL_NUMS'] = [1, 2, 4, 8, 16, 32]
    
    return params
