import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import load_global_params

global_params = load_global_params()

# Emission centre wavelength (nm) of each output channel, used as attention input.
CHAN_NM = torch.tensor(global_params["CHAN_WAVELENGTHS_ARRAY"], dtype=torch.float32)


# -----------------------------------------------------------------------------
# Phasor loss
# -----------------------------------------------------------------------------
def phasor_components(spectrum, harmonic=None):
    """Spectral phasor coordinates (G, S) of each spectrum along the last dimension."""
    if harmonic is None:
        harmonic = global_params["DEFAULT_HARMONIC"]

    N = spectrum.shape[-1]
    idx = torch.arange(N, device=spectrum.device).float()
    angles = 2 * np.pi * harmonic * idx / N
    sum_intensity = spectrum.sum(dim=-1) + global_params["PHASOR_EPSILON"]

    G = (spectrum * torch.cos(angles)).sum(dim=-1) / sum_intensity
    S = (spectrum * torch.sin(angles)).sum(dim=-1) / sum_intensity
    return G, S


def phasor_loss(pred_spectrum, true_spectrum, harmonic=None):
    """Mean Euclidean distance between predicted and true spectra in the phasor plane."""
    if harmonic is None:
        harmonic = global_params["DEFAULT_HARMONIC"]
    G_pred, S_pred = phasor_components(pred_spectrum, harmonic)
    G_true, S_true = phasor_components(true_spectrum, harmonic)
    return torch.sqrt((G_pred - G_true) ** 2 + (S_pred - S_true) ** 2).mean()


def combined_spectral_loss(pred32, target32, input_x, lambda_consistency=None, lambda_phasor=None, harmonic=None):
    """MSE + lambda_consistency * consistency + lambda_phasor * phasor distance.

    The consistency term compares the prediction, averaged over each group of
    GROUP_SIZE adjacent output channels, with the input channels averaged over
    repeats.

    pred32, target32: (B, 32)
    input_x:          (B, 1, channel_num, R)
    """
    if lambda_consistency is None:
        lambda_consistency = global_params["LAMBDA_CONSISTENCY"]
    if lambda_phasor is None:
        lambda_phasor = global_params["LAMBDA_PHASOR"]
    if harmonic is None:
        harmonic = global_params["DEFAULT_HARMONIC"]

    channel_num = global_params["channel_num"]
    group_size = global_params["GROUP_SIZE"]

    mse32 = F.mse_loss(pred32, target32)

    pred_groups = pred32.view(pred32.size(0), channel_num, group_size).mean(dim=2)  # (B, channel_num)
    input_mean = input_x.squeeze(1).mean(dim=2)                                      # (B, channel_num)
    consistency = F.mse_loss(pred_groups, input_mean)

    phasor = phasor_loss(pred32, target32, harmonic=harmonic)
    return mse32 + lambda_consistency * consistency + lambda_phasor * phasor


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------
class ResidualBlock1D(nn.Module):
    """Conv1d → BN → LeakyReLU → Conv1d → BN, plus identity skip, then LeakyReLU."""
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        kernel_size = global_params["KERNEL_SIZE"]
        alpha = global_params["LEAKY_RELU_ALPHA"]
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + x)


class ResidualBlock2D(nn.Module):
    """Conv2d → BN → LeakyReLU → Conv2d → BN, plus identity skip, then LeakyReLU."""
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        kernel_size = global_params["KERNEL_SIZE"]
        alpha = global_params["LEAKY_RELU_ALPHA"]
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + x)


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------
class PRESD(nn.Module):
    """PRESD network: reconstructs a 32-channel emission spectrum for each pixel
    from C_in reduced spectral channels acquired R times.

    Shapes are per pixel (batch dimension omitted). With the default
    configuration: C_in = channel_num, 221 = BASE_CHANNELS, 86 = DECODER_CHANNELS.

    (a) Convolutional feature extraction
        reduced spectral input                                   C_in x R
        Conv2d 3x3 (pad 1) + BN + LeakyReLU                      221 x C_in x R
        concat excitation wavelength λ_ex as an extra channel    222 x C_in x R
    (b) Residual encoding & repeat integration
        5 x ResidualBlock2D (two 3x3 convs each)                 222 x C_in x R
        transition FC (linear R → 1)                             222 x C_in
    (c) Wavelength-aware self-attention
        ConvTranspose1d (C_in → 32, 222 → 86)                    86 x 32
        append λ_ex and λ_em rows to each of the 32 tokens       88 x 32
        multi-head self-attention (4 heads, embed 88) + residual,
        then keep the 86 feature rows                            86 x 32
    (d) Spectral reconstruction
        BN + LeakyReLU + ResidualBlock1D (two 3-tap convs)       86 x 32
        Conv1d 1x1 (project to 1)                                1 x 32
        reconstructed emission spectrum                          32
    """
    def __init__(self, base_channels=None, replicate_count=None):
        super().__init__()
        params = load_global_params()

        if replicate_count is None:
            raise ValueError("replicate_count (number of repeats R) must be provided.")
        if base_channels is None:
            base_channels = params["BASE_CHANNELS"]

        channel_num = params["channel_num"]
        alpha = params["LEAKY_RELU_ALPHA"]
        decoder_channels = params["DECODER_CHANNELS"]
        self.output_channels = params["OUTPUT_CHANNELS"]
        self.decoder_channels = decoder_channels

        # (a) Convolutional feature extraction
        self.entry = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=params["KERNEL_SIZE"], padding=params["PADDING"]),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(alpha, inplace=True),
        )

        # (b) Residual encoding & repeat integration
        self.blocks = nn.ModuleList([
            ResidualBlock2D(base_channels + 1, dilation=1)
            for _ in range(params["NUM_BLOCKS"])
        ])
        self.transition_fc = nn.Linear(replicate_count, 1)

        # (c) Wavelength-aware self-attention
        # Transposed conv: C_in → 32 along the spectral axis, 222 → 86 features.
        # (kernel_size, padding) are chosen so the output length is exactly 32.
        upsampling_factor = self.output_channels // channel_num
        upsample_configs = {2: (4, 1), 4: (8, 2), 8: (16, 4)}
        kernel_size, padding = upsample_configs.get(
            upsampling_factor, (upsampling_factor * 2, upsampling_factor // 2))
        self.up = nn.ConvTranspose1d(base_channels + 1, decoder_channels,
                                     kernel_size=kernel_size, stride=upsampling_factor, padding=padding)
        # Token features: 86 learned features + λ_ex + λ_em = 88.
        self.final_attn = nn.MultiheadAttention(embed_dim=decoder_channels + 2,
                                                num_heads=params["FINAL_ATTN_HEADS"])

        # (d) Spectral reconstruction
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(decoder_channels),
            nn.LeakyReLU(alpha, inplace=True),
            ResidualBlock1D(decoder_channels, dilation=1),
            nn.Conv1d(decoder_channels, 1, kernel_size=params["FINAL_KERNEL_SIZE"]),
        )

        print(f"PRESD initialized • repeats={replicate_count} • channels={channel_num}")

    def forward(self, x, wave=None):
        """
        Args:
            x:    (B, 1, C_in, R) or (B, R, C_in), normalized reduced spectral input
            wave: (B,) excitation wavelength λ_ex in nm, or None to skip wavelength conditioning
        Returns:
            (B, 32) reconstructed spectrum (same normalization as the training targets)
        """
        if x.dim() == 3:
            x = x.permute(0, 2, 1).unsqueeze(1)                     # (B, 1, C_in, R)

        # (a) Convolutional feature extraction
        z = self.entry(x)                                            # (B, 221, C_in, R)
        if wave is not None:
            B, _, H, W = z.shape
            ex_channel = wave.float().view(B, 1, 1, 1).to(z.device).expand(B, 1, H, W)
            z = torch.cat([z, ex_channel], dim=1)                    # (B, 222, C_in, R)

        # (b) Residual encoding & repeat integration
        for blk in self.blocks:
            z = blk(z)                                               # (B, 222, C_in, R)
        z = self.transition_fc(z).squeeze(-1)                        # (B, 222, C_in)

        # (c) Wavelength-aware self-attention
        z = self.up(z)                                               # (B, 86, 32)
        if wave is not None:
            B = z.size(0)
            lambda_ex = wave.float().unsqueeze(-1).repeat(1, self.output_channels).to(z.device)  # (B, 32)
            lambda_em = CHAN_NM.to(z.device).unsqueeze(0).repeat(B, 1)                            # (B, 32)
            tokens = torch.cat([z, torch.stack([lambda_ex, lambda_em], dim=1)], dim=1)            # (B, 88, 32)
            seq = tokens.permute(2, 0, 1)                            # (32, B, 88): 32 spectral tokens
            attn_out, _ = self.final_attn(seq, seq, seq)
            z = (tokens + attn_out.permute(1, 2, 0))[:, :self.decoder_channels, :]  # (B, 86, 32)

        # (d) Spectral reconstruction
        return self.decoder(z).squeeze(1)                            # (B, 32)
