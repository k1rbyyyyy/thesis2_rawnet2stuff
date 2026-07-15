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
            print(f"No protocol file found. Automatically indexing files in {data_dir}...")
            # Support both .flac and .wav
            all_files = [f for f in os.listdir(data_dir) if f.endswith('.wav') or f.endswith('.flac')]
            self.file_list = [os.path.join(data_dir, f) for f in all_files]
            self.labels = [0] * len(self.file_list)
        
        # Case B: Standard ASVspoof Protocol File (Actual Training phase)
        else:
            print(f"Loading files listed in protocol: {protocol_file}")
            with open(protocol_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 4:
                    base_name = parts[1]
                    label_str = parts[-1]
                    
                    # Try finding .flac first, then fallback to .wav
                    file_path = os.path.join(data_dir, base_name + ".flac")
                    if not os.path.exists(file_path):
                        file_path = os.path.join(data_dir, base_name + ".wav")
                        
                    if os.path.exists(file_path):
                        self.file_list.append(file_path)
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
    

class ASVspoof2021EvalDataset(Dataset):
    """
    Custom PyTorch Dataset for loading ASVspoof 2021 Evaluation Track data.
    Reads .flac or .wav evaluation files and returns the waveform along with the filename string.
    """
    def __init__(self, data_dir, protocol_file, max_samples=64000):
        """
        Args:
            data_dir (str): Directory containing the evaluation audio files.
            protocol_file (str): Path to the evaluation trial protocol file (e.g., keys/metadata).
            max_samples (int): Max sample count for RawNet2 (64,000 samples).
        """
        self.data_dir = data_dir
        self.max_samples = max_samples
        self.file_list = []

        print(f"Loading 2021 Evaluation files listed in: {protocol_file}")
        with open(protocol_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                # Standard 2021 LA eval protocol line format usually provides the filename
                # Adjust index if your specific evaluation protocol format differs
                file_name = parts[1] 
                self.file_list.append(file_name)

        print(f"Evaluation Dataset initialization complete. Found {len(self.file_list)} trial entries.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        
        # Check for .flac first (2021 standard), fallback to .wav if you converted them
        file_path = os.path.join(self.data_dir, f"{file_name}.flac")
        if not os.path.exists(file_path):
            file_path = os.path.join(self.data_dir, f"{file_name}.wav")

        # 1. Load the raw audio file
        x, sample_rate = sf.read(file_path)

        # Ensure mono channel format
        if len(x.shape) > 1:
            x = np.mean(x, axis=1)

        # 2. Fix audio length: Cut or Pad to precisely match expected samples (64,000)
        x_len = len(x)
        if x_len < self.max_samples:
            # Pad with zeros if the clip is too short
            difference = self.max_samples - x_len
            x = np.pad(x, (0, difference), 'constant')
        else:
            # Symmetrical or standard crop for evaluation consistency (avoiding random crop during eval)
            start_idx = (x_len - self.max_samples) // 2
            x = x[start_idx : start_idx + self.max_samples]

        # Convert to torch Tensor float32
        x = torch.from_numpy(x).float()
        
        # Return the waveform and filename string (essential for EER generation script)
        return x, file_name
    
    