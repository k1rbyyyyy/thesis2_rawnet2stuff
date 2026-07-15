import os
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

# Import your own modules (assuming model.py and dataset.py are in your directory)
from model import RawNet2
from dataset import RawNet2Dataset

# ==========================================
# STEP 1: CREATE DUMMY AUDIO FILES LOCALLY
# ==========================================
temp_dir = "./temp_test_audio"
os.makedirs(temp_dir, exist_ok=True)

print("Creating 5 dummy .wav files to simulate a dataset...")
for i in range(5):
    # Create 3 seconds of random noise at 16kHz
    dummy_noise = np.random.randn(16000 * 3) 
    file_path = os.path.join(temp_dir, f"dummy_audio_{i}.wav")
    sf.write(file_path, dummy_noise, 16000)
print("Dummy files generated!")

# ==========================================
# STEP 2: TEST THE DATA LOADER
# ==========================================
# Load our custom dataset targeting our temp folder
test_dataset = RawNet2Dataset(data_dir=temp_dir, protocol_file=None, max_samples=64000)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)

# Pull one batch out of the loader
audio_batch, label_batch = next(iter(test_loader))
print(f"\nBatch Loaded successfully!")
print(f"-> Audio Batch Shape: {audio_batch.shape} (Expected: [batch_size, 64000])")
print(f"-> Labels: {label_batch}")

# ==========================================
# STEP 3: PASS BATCH THROUGH THE RAWNET2 MODEL
# ==========================================
print("\nInitializing RawNet2 model...")
model = RawNet2(out_classes=2)

print("Passing audio batch through the RawNet2 network...")
model.eval()
with torch.no_grad():
    predictions = model(audio_batch)

print("\n--- Pipeline Check Complete! ---")
print(f"Output Raw Logits shape: {predictions.shape}")
print(f"Logits: {predictions}")