import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf

class RawNet2Dataset(Dataset):
    """
    Custom PyTorch Dataset for loading raw waveforms for RawNet2.
    It reads .wav files and ensures they are cut or padded to a fixed sample length.
    """
    def __init__(self, data_dir, protocol_file=None, max_samples=64000, is_test_run=False):
        """
        Args:
            data_dir (str): Directory containing the actual .wav files.
            protocol_file (str): Path to the ASVspoof protocol txt file (e.g., LA.txt).
                                If None, it will automatically search and load all .wav files 
                                in data_dir (perfect for local testing without protocols!).
            max_samples (int): Max sample count. 64,000 samples = 4 seconds of 16kHz audio.
            is_test_run (bool): If True, automatically generates dummy targets.
        """
        self.data_dir = data_dir
        self.max_samples = max_samples
        self.file_list = []
        self.labels = []

        # Case A: No protocol file provided (Local prototyping phase)
        if protocol_file is None or not os.path.exists(protocol_file):
            print(f"No protocol file found. Automatically indexing all .wav files in {data_dir}...")
            all_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
            self.file_list = [os.path.join(data_dir, f) for f in all_files]
            # Default everything to 'bonafide' (0) for testing structure
            self.labels = [0] * len(self.file_list)
        
        # Case B: Standard ASVspoof Protocol File (Actual Training phase)
        else:
            print(f"Loading files listed in protocol: {protocol_file}")
            with open(protocol_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 4:
                    # Typical ASVspoof format: [Speaker_ID, Audio_File_Name, -, -, Label]
                    # Example: LA_0079 LA_T_1132912 - - bonafide
                    file_name = parts[1] + ".wav"
                    label_str = parts[-1]
                    
                    file_path = os.path.join(data_dir, file_name)
                    if os.path.exists(file_path):
                        self.file_list.append(file_path)
                        # 0 for genuine (bonafide), 1 for spoofed (fake)
                        self.labels.append(0 if label_str == 'bonafide' else 1)

        print(f"Dataset initialization complete. Found {len(self.file_list)} valid audio samples.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx]

        # 1. Load the raw audio file
        # RawNet2 processes raw waveforms directly, skipping spectrogram creation!
        x, sample_rate = sf.read(file_path)

        # Ensure mono channel format
        if len(x.shape) > 1:
            x = np.mean(x, axis=1)

        # 2. Fix audio length: Cut or Pad to precisely match expected samples (e.g. 64,000)
        x_len = len(x)
        if x_len < self.max_samples:
            # Pad with zeros if the clip is too short
            difference = self.max_samples - x_len
            x = np.pad(x, (0, difference), 'constant')
        else:
            # Random crop if it's too long (helps with data regularization)
            start_idx = np.random.randint(0, x_len - self.max_samples + 1)
            x = x[start_idx : start_idx + self.max_samples]

        # Convert to torch Tensor float32
        x = torch.from_numpy(x).float()
        
        return x, label