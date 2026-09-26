import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------------------------------------
#  Configuration ----------------------------------------------------------------
# -----------------------------------------------------------------------------
from config import load_global_params

# Load configuration
global_params = load_global_params()
channel_num = global_params["channel_num"]

# 32-channel emission wavelengths for wavelength-aware attention
CHAN_NM = torch.tensor(global_params["CHAN_WAVELENGTHS_ARRAY"], dtype=torch.float32)

# -----------------------------------------------------------------------------
#  Phasor loss functions -------------------------------------------------------
# -----------------------------------------------------------------------------
def phasor_components(spectrum, harmonic=None):
    """Calculate G and S phasor components at a given harmonic."""
    if harmonic is None:
        harmonic = global_params["DEFAULT_HARMONIC"]
    
    N = spectrum.shape[-1]
    device = spectrum.device
    idx = torch.arange(N, device=device).float()

    angles = 2 * np.pi * harmonic * idx / N
    sum_intensity = spectrum.sum(dim=-1, keepdim=True) + global_params["PHASOR_EPSILON"]

    G = (spectrum * torch.cos(angles)).sum(dim=-1) / sum_intensity.squeeze(-1)
    S = (spectrum * torch.sin(angles)).sum(dim=-1) / sum_intensity.squeeze(-1)

    return G, S

def phasor_loss(pred_spectrum, true_spectrum, harmonic=None):
    """Compute Euclidean distance in the phasor domain between predicted and ground truth spectra."""
    if harmonic is None:
        harmonic = global_params["DEFAULT_HARMONIC"]
        
    G_pred, S_pred = phasor_components(pred_spectrum, harmonic)
    G_true, S_true = phasor_components(true_spectrum, harmonic)
    return torch.sqrt((G_pred - G_true)**2 + (S_pred - S_true)**2).mean()

# -----------------------------------------------------------------------------
#  Helper blocks ----------------------------------------------------------------
# -----------------------------------------------------------------------------
class ResidualBlock1D(nn.Module):
    """A simple 1‑D residual block: Conv → BN → LeakyReLU → Conv → BN + skip."""
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        kernel_size = global_params["KERNEL_SIZE"]
        alpha = global_params["LEAKY_RELU_ALPHA"]
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, 
                              padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, 
                              padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)

class ResidualBlock2D(nn.Module):
    """A simple 2‑D residual block: Conv2d → BN → LeakyReLU → Conv2d → BN + skip."""
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        kernel_size = global_params["KERNEL_SIZE"]
        alpha = global_params["LEAKY_RELU_ALPHA"]
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, 
                              padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, 
                              padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)

# -----------------------------------------------------------------------------
#  Physics‑aware loss ----------------------------------------------------------
# -----------------------------------------------------------------------------
def combined_spectral_loss(pred32, target32, input16xR, lambda_consistency=None, lambda_phasor=None, harmonic=None):
    """Combined reconstruction + consistency + phasor loss."""
    if lambda_consistency is None:
        lambda_consistency = global_params["LAMBDA_CONSISTENCY"]
    if lambda_phasor is None:
        lambda_phasor = global_params["LAMBDA_PHASOR"]
    if harmonic is None:
        harmonic = global_params["DEFAULT_HARMONIC"]
    
    # MSE + consistency loss
    mse32 = F.mse_loss(pred32, target32)
    
    # Consistency: channel groups in output should match input channels
    group_size = global_params["GROUP_SIZE"]
    channel_groups = pred32.view(pred32.size(0), channel_num, group_size).mean(dim=2)
    
    # Get mean of input channels across replicates
    if input16xR.dim() == 4:
        input16xR = input16xR.squeeze(1).permute(0, 2, 1)
    input_mean = input16xR.mean(dim=1)
    
    consistency_loss = F.mse_loss(channel_groups, input_mean)
    mse_loss = mse32 + lambda_consistency * consistency_loss
        
    phasor = phasor_loss(pred32, target32, harmonic=harmonic)
    return mse_loss + lambda_phasor * phasor 

class FilmEarlyNet(nn.Module):
    """
    FilmEarlyNet with Feature-wise Linear Modulation (FiLM) and wavelength-aware attention.
    
    Key features:
    - Takes both spectral data (x) and wavelength information (wave) as inputs
    - Uses attention mechanism for wavelength-aware processing  
    - Adds laser wavelength as an additional channel for conditioning
    """
    def __init__(self, base_channels=None, num_heads=None, replicate_count=None, **kwargs):
        super().__init__()
        params = load_global_params()
        
        # replicate_count must be explicitly provided
        if replicate_count is None:
            raise ValueError("replicate_count must be explicitly provided to FilmEarlyNet constructor. "
                           "This defines the number of input channels (number of position repeats).")
        
        print(f"FilmEarlyNet (Hybrid 2D with FC transition) initializing with replicate_count: {replicate_count}")

        # Use global parameters with optional overrides
        if base_channels is None:
            base_channels = params["BASE_CHANNELS"]
        if num_heads is None:
            num_heads = params["NUM_HEADS"]

        # Entry conv - 2D version
        kernel_size = params["KERNEL_SIZE"]
        padding = params["PADDING"]
        alpha = params["LEAKY_RELU_ALPHA"]
        
        self.entry = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(alpha, inplace=True),
        )

        # Residual blocks - 2D version for base_channels + 1 laser channel
        num_blocks = params.get('NUM_BLOCKS', params["NUM_BLOCKS"])
        self.blocks = nn.ModuleList([
            ResidualBlock2D(base_channels + 1, dilation=1)
            for _ in range(num_blocks)
        ])

        # Transition FC layer: reduces R dimension (B, 65, channel_num, R) → (B, 65, channel_num, 1)
        self.transition_fc = nn.Linear(replicate_count, 1)

        # Simplified upsampling configuration
        channel_num = params['channel_num']
        upsampling_factor = params["OUTPUT_CHANNELS"] // channel_num
        upsample_configs = {
            2: (4, 1), 4: (8, 2), 8: (16, 4)
        }
        kernel_size, padding = upsample_configs.get(upsampling_factor, (upsampling_factor * 2, upsampling_factor // 2))
        
        decoder_channels = params["DECODER_CHANNELS"]
        self.up = nn.ConvTranspose1d(base_channels + 1, decoder_channels, 
                                    kernel_size=kernel_size, stride=upsampling_factor, padding=padding)
        
        # Decoder
        final_kernel_size = params["FINAL_KERNEL_SIZE"]
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(decoder_channels),
            nn.LeakyReLU(alpha, inplace=True),
            ResidualBlock1D(decoder_channels, dilation=1),
            nn.Conv1d(decoder_channels, 1, kernel_size=final_kernel_size),
        )
        
        # Final attention layer
        # Input to attention is z (decoder_channels) + wavelength_matrix (2) = decoder_channels + 2
        attn_embed_dim = decoder_channels + 2
        final_attn_heads = params["FINAL_ATTN_HEADS"]
        self.final_attn = nn.MultiheadAttention(embed_dim=attn_embed_dim, num_heads=final_attn_heads)
        
        # Store parameters needed in forward pass as instance variables
        # (to avoid mismatch when global_params is updated between trials)
        self.decoder_channels = decoder_channels
        self.output_channels = params["OUTPUT_CHANNELS"]

        print(f"FilmEarlyNet (Hybrid 2D→1D with FC transition) initialized • replicates={replicate_count} • channels={channel_num} • attention_heads={num_heads}")

    def forward(self, x, wave=None):
        """
        Forward pass with wavelength conditioning and attention (Hybrid 2D/1D with FC transition).
        
        Args:
            x: Input spectral data (B, 1, channel_num, R) or (B, R, channel_num)
            wave: Wavelength information (B,) - normalized wavelength values
        
        Returns:
            Output spectrum (B, 32)
        """
        # Ensure input is 4D for 2D processing: (B, 1, channel_num, R)
        if x.dim() == 3:
            # (B, R, channel_num) → (B, 1, channel_num, R)
            x = x.permute(0, 2, 1).unsqueeze(1)
        elif x.dim() == 4 and x.size(1) != 1:
            # Ensure channel is first
            x = x.permute(0, 1, 3, 2) if x.size(1) > 1 else x
            
        z = self.entry(x)  # (B, 64, channel_num, R)

        # Add laser wavelength channel as 2D feature map
        if wave is not None:
            laser_param = wave.float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(z.device)
            B, C, H, W = z.shape
            laser_channel = laser_param.expand(B, 1, H, W)
            z = torch.cat([z, laser_channel], dim=1)  # (B, 65, channel_num, R)

        # Apply 2D residual blocks
        for blk in self.blocks:
            z = blk(z)  # (B, 65, channel_num, R)

        # Transition from 2D to 1D: use FC layer to reduce R dimension
        # (B, 65, channel_num, R) → (B, 65, channel_num, 1) → (B, 65, channel_num)
        z = self.transition_fc(z).squeeze(-1)  # (B, 65, channel_num)

        z = self.up(z)  # (B, 64, 32)
        
        # Create wavelength matrix for attention
        if wave is not None:
            B = z.size(0)
            laser_wavelengths = wave.float().unsqueeze(-1).repeat(1, self.output_channels).to(z.device)
            chan_wavelengths = (CHAN_NM.to(z.device) / 1.0).unsqueeze(0).repeat(B, 1)
            wavelength_matrix = torch.stack([laser_wavelengths, chan_wavelengths], dim=1)
            
            # Apply attention
            z_with_wavelengths = torch.cat([z, wavelength_matrix], dim=1)
            seq_32, _ = self.final_attn(z_with_wavelengths.permute(2,0,1), z_with_wavelengths.permute(2,0,1), z_with_wavelengths.permute(2,0,1))
            seq_32 = seq_32.permute(1,2,0)
            
            z_attended = z_with_wavelengths + seq_32
            z_final = z_attended[:, :self.decoder_channels, :]  # Extract feature channels
        else:
            z_final = z
        
        return self.decoder(z_final).squeeze(1) 