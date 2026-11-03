"""
Convolutional Recurrent Network (CRN) for Complex Spectral Speech Enhancement & Denoising.
Supports both real and complex STFT representations with encoder-LSTM-decoder U-Net topology.
"""
from __future__ import annotations
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ComplexConv2d(nn.Module):
        """Complex 2D Convolution layer."""
        def __init__(self, in_channels: int, out_channels: int, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)):
            super().__init__()
            self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        def forward(self, xr: torch.Tensor, xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            real = self.conv_r(xr) - self.conv_i(xi)
            imag = self.conv_r(xi) + self.conv_i(xr)
            return real, imag

    class CRN(nn.Module):
        """
        Convolutional Recurrent Network for Speech Enhancement.
        Input: (Batch, 2, Frames, Frequency_Bins) -> Real and Imaginary STFT
        Output: (Batch, 2, Frames, Frequency_Bins) -> Estimated Clean STFT
        """
        def __init__(self, n_fft: int = 512, hop_length: int = 256):
            super().__init__()
            self.n_fft = n_fft
            self.hop_length = hop_length
            freq_bins = n_fft // 2 + 1  # 257

            # Encoder
            self.enc1 = nn.Conv2d(2, 16, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
            self.bn1 = nn.BatchNorm2d(16)
            self.enc2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
            self.bn2 = nn.BatchNorm2d(32)
            self.enc3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
            self.bn3 = nn.BatchNorm2d(64)

            # Bottleneck LSTM
            # 257 -> 129 -> 65 -> 33
            self.lstm = nn.LSTM(input_size=64 * 33, hidden_size=256, num_layers=2, batch_first=True)
            self.fc = nn.Linear(256, 64 * 33)

            # Decoder (Transposed Convolutions)
            self.dec3 = nn.ConvTranspose2d(128, 32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), output_padding=(0, 1))
            self.dbn3 = nn.BatchNorm2d(32)
            self.dec2 = nn.ConvTranspose2d(64, 16, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), output_padding=(0, 1))
            self.dbn2 = nn.BatchNorm2d(16)
            self.dec1 = nn.ConvTranspose2d(32, 2, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1), output_padding=(0, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (Batch, 2, T, F) STFT features (channel 0: Real, channel 1: Imag)
            Returns:
                mask: (Batch, 2, T, F) Complex ratio mask or direct spectral estimate
            """
            b, c, t, f = x.shape

            # Encoder path
            e1 = F.elu(self.bn1(self.enc1(x)))
            e2 = F.elu(self.bn2(self.enc2(e1)))
            e3 = F.elu(self.bn3(self.enc3(e2)))

            # Bottleneck
            b_size, ch, frames, freq = e3.shape
            lstm_in = e3.permute(0, 2, 1, 3).contiguous().view(b_size, frames, -1)
            lstm_out, _ = self.lstm(lstm_in)
            fc_out = self.fc(lstm_out)
            d_in = fc_out.view(b_size, frames, ch, freq).permute(0, 2, 1, 3)

            # Decoder path with skip connections
            d3 = F.elu(self.dbn3(self.dec3(torch.cat([d_in, e3], dim=1))))
            d2 = F.elu(self.dbn2(self.dec2(torch.cat([d3, e2], dim=1))))
            out = self.dec1(torch.cat([d2, e1], dim=1))

            # Crop or pad to match exact input frequency dimension
            if out.shape[-1] > f:
                out = out[:, :, :, :f]
            elif out.shape[-1] < f:
                out = F.pad(out, (0, f - out.shape[-1]))

            return torch.tanh(out)

except ImportError:
    # Pure numpy fallback placeholder
    class CRN:
        def __init__(self, n_fft: int = 512, hop_length: int = 256):
            self.n_fft = n_fft
            self.hop_length = hop_length

        def __call__(self, x: np.ndarray) -> np.ndarray:
            # Pass-through or Wiener filter mask simulation
            return np.tanh(x * 0.9)
