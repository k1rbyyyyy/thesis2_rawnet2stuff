import os
import numpy as np
import torch
from torch.utils.data import Dataset
import soundfile as sf

class ASVspoof2021EvalDataset(Dataset):
    """
    Custom PyTorch Dataset optimized for loading raw waveforms 
    from the ASVspoof 2021 LA Evaluation partition.
    """
    def __init__(self, data_dir, protocol_file, max_samples=64000):
        """
        Args:
            data_dir (str): Path to the folder containing the 2021 FLAC/WAV files.
            protocol_file (str): Path to the 2021 CM (Countermeasure) evaluation key file.
            max_samples (int): 64,000 samples = 4 seconds of 16kHz audio.
        """
        self.data_dir = data_dir
        self.max_samples = max_samples
        self.file_list = []
        self.labels = []

        print(f"Reading ASVspoof 2021 Evaluation keys from: {protocol_file}")
        
        # Read the 2021 evaluation keys
        with open(protocol_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            # ASVspoof 2021 LA Eval Metadata structure:
            # [LA_0023, LA_E_2013854, tele, codec, vocoder, ..., bonafide/spoof]
            if len(parts) >= 2:
                file_name = parts[1] # e.g. "LA_E_2013854"
                label_str = parts[-1] # The last item is "bonafide" or "spoof"
                
                # Check for either .flac or .wav extension depending on your download format
                file_path_wav = os.path.join(data_dir, f"{file_name}.wav")
                file_path_flac = os.path.join(data_dir, f"{file_name}.flac")
                
                if os.path.exists(file_path_wav):
                    self.file_list.append(file_path_wav)
                    self.labels.append(0 if label_str == 'bonafide' else 1)
                elif os.path.exists(file_path_flac):
                    self.file_list.append(file_path_flac)
                    self.labels.append(0 if label_str == 'bonafide' else 1)

        print(f"Successfully loaded {len(self.file_list)} files from the 2021 Evaluation set.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx]

        # Load audio (soundfile natively reads both FLAC and WAV)
        x, sample_rate = sf.read(file_path)

        # Force mono
        if len(x.shape) > 1:
            x = np.mean(x, axis=1)

        # Fix length to exactly max_samples (default 4 seconds at 16kHz)
        x_len = len(x)
        if x_len < self.max_samples:
            difference = self.max_samples - x_len
            x = np.pad(x, (0, difference), 'constant')
        else:
            # Crop middle segment to keep the core audio characteristics consistent
            start_idx = (x_len - self.max_samples) // 2
            x = x[start_idx : start_idx + self.max_samples]

        # Convert to Tensor
        x = torch.from_numpy(x).float()
        return x, label