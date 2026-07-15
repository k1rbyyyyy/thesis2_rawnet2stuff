import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. THE SINCCONV FRONT-END LAYER
# ==========================================
class SincConv(nn.Module):
    """
    Sinc-convolution layer that learns band-pass filter boundaries.
    Adapted for standard PyTorch execution without custom CUDA dependencies.
    """
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, min_low_hz=50, min_band_hz=50):
        super(SincConv, self).__init__()
        
        if kernel_size % 2 == 0:
            raise ValueError("SincConv requires an odd kernel_size.")
            
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # Initialize filter bank frequencies using Mel scale distribution
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)
        
        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1)
        hz = self.to_hz(mel)
        
        # Trainable parameters: low frequency cutoffs and frequency bandwidths
        self.low_hz_ = nn.Parameter(torch.from_numpy(hz[:-1]).float().view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.from_numpy(np.diff(hz)).float().view(-1, 1))

        # Ham Window generation
        n_lin = torch.linspace(0, (self.kernel_size - 1) / 2, steps=int((self.kernel_size + 1) / 2))
        self.register_buffer('window_', 0.54 - 0.46 * torch.cos(2 * np.pi * n_lin / self.kernel_size))
        
        # Time matrix for the sinc function calculation
        n_ = 2 * np.pi * torch.arange(-((self.kernel_size - 1) / 2), 0).view(1, -1) / self.sample_rate
        self.register_buffer('n_', n_)

    def forward(self, waveforms):
        # Calculate cutoffs enforcing minimum physical bounds
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]
        
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        # Standard Sinc function formulation: sin(x)/x
        # Left half of the symmetric filter + center point
        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)) * self.window_[:-1]
        band_pass_center = (2 * band).view(-1, 1) * self.window_[-1]
        
        # Reconstruct full symmetric filters
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        filters = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        filters = filters / (2 * band[:, None]) # Normalize filter energy

        # Format filters to match standard Conv1d expected shape: (out_channels, in_channels, kernel_size)
        return F.conv1d(waveforms, filters.view(self.out_channels, 1, self.kernel_size), stride=1, padding=self.kernel_size // 2)


# ==========================================
# 2. FEATURE MAP SCALING (FMS) BLOCK
# ==========================================
class FMS(nn.Module):
    """
    Feature Map Scaling block acts as an attention layer.
    It weights channels dynamically based on global contexts.
    """
    def __init__(self, channels):
        super(FMS, self).__init__()
        self.fc = nn.Linear(channels, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, channels, timesteps)
        w = F.adaptive_avg_pool1d(x, 1).squeeze(-1) # Global average pooling
        w = self.fc(w)
        w = self.sigmoid(w).unsqueeze(-1) # Scale to shape (batch, channels, 1)
        return x * w


# ==========================================
# 3. RESIDUAL BLOCK WITH FMS
# ==========================================
class ResidualBlock(nn.Module):
    """
    Standard 1D residual connection combined with a post-processing FMS block
    and an architectural MaxPool1d layer for temporal downsampling.
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.3)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        
        self.fms = FMS(out_channels)
        
        # Max pooling downsampling layer added to every block
        self.mp = nn.MaxPool1d(kernel_size=3)
        
        # Shortcut connection handling dimension mismatches
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.bn1(x)
        out = self.leaky_relu(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        
        out = self.fms(out)
        
        # Element-wise addition occurs BEFORE the max pooling operation
        out = out + residual
        out = self.mp(out)
        
        return out


# ==========================================
# 4. THE COMPLETE RAWNET2 ARCHITECTURE
# ==========================================
class RawNet2(nn.Module):
    """
    RawNet2 end-to-end network for raw waveform classification.
    Expected input: Waveforms of shape (batch_size, 1, sequence_length)
    """
    def __init__(self, out_classes=2):
        super(RawNet2, self).__init__()
        
        # 1. Frontend trainable Sinc-convolution
        self.sinc_conv = SincConv(out_channels=128, kernel_size=251) # 128 filters
        self.first_bn = nn.BatchNorm1d(128)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.3)
        self.max_pool = nn.MaxPool1d(kernel_size=3) # Pooling down steps
        
        # 2. Sequential Residual blocks
        # In RawNet2, we start with 128 channels and keep them aligned.
        self.block1 = ResidualBlock(128, 128)
        self.block2 = ResidualBlock(128, 128)
        self.block3 = ResidualBlock(128, 512) # Reshape channel width 
        self.block4 = ResidualBlock(512, 512)
        self.block5 = ResidualBlock(512, 512)
        self.block6 = ResidualBlock(512, 512)
        
        self.bn_before_gru = nn.BatchNorm1d(512)
        
        # 3. Recurrent Layer (GRU) processing sequences
        # GRU expects shape: (batch, sequence_length, features)
        self.gru = nn.GRU(input_size=512, hidden_size=1024, num_layers=3, batch_first=True)
        
        # 4. Dense classification layer
        self.fc_classifier = nn.Linear(1024, out_classes)

    def forward(self, x):
        # Ensure correct channel configuration: (batch, 1, length)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # Frontend features (Runs on GPU via DirectML)
        x = self.sinc_conv(x)
        x = self.first_bn(x)
        x = self.leaky_relu(x)
        x = self.max_pool(x)
        
        # Residual extraction blocks (Runs on GPU via DirectML)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        
        x = self.bn_before_gru(x)
        x = self.leaky_relu(x)
        
        # Prepare for sequence processing: transpose (batch, channels, timesteps) -> (batch, timesteps, channels)
        x = x.transpose(1, 2)
        
        # =====================================================================
        # DYNAMIC DEVICE BRIDGE WORKAROUND FOR DIRECTML RNN FALLBACK BUG
        # =====================================================================
        # 1. Determine where the GRU parameters are located (could be GPU or CPU)
        gru_device = next(self.gru.parameters()).device
        
        # 2. If the current tensor is on the GPU but the GRU is on the CPU, copy it over
        if x.device != gru_device:
            x = x.to(gru_device)
            
        # GRU returns (out, h_n) where h_n contains final hidden state (Runs on CPU)
        _, h_n = self.gru(x)
        
        # We take the output of the final GRU layer 
        feat = h_n[-1] # shape: (batch, 1024)
        
        # 3. Safely copy the feature tensor back to match the FC classifier's device (GPU)
        fc_device = self.fc_classifier.weight.device
        if feat.device != fc_device:
            feat = feat.to(fc_device)
        # =====================================================================
        
        out = self.fc_classifier(feat)
        return out


# ==========================================
# TEST IMPLEMENTATION BLOCK
# ==========================================
if __name__ == "__main__":
    print("Initializing RawNet2 module...")
    model = RawNet2(out_classes=2)
    
    # Simulate a single batch of audio: 1 file, 4 seconds of 16kHz audio = 64,000 samples
    simulated_batch = torch.randn(1, 1, 64000)
    
    print(f"Feeding audio shape {simulated_batch.shape} through model...")
    with torch.no_grad():
        output = model(simulated_batch)
        
    print("Feed-forward successful!")
    print(f"Classification shape output (Logits): {output.shape} -> {output.numpy()}")