import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.adam import Adam

# 1. Device Auto-Detection (DirectML Windows / Standard CUDA / CPU)
def get_optimal_device():
    # Check for Windows DirectML (Radeon RX 6600)
    try:
        import torch_directml
        if torch_directml.is_available():
            print("--> AMD GPU Detected via DirectML!")
            return torch_directml.device()
    except ImportError:
        pass
    
    # Check for NVIDIA CUDA or AMD ROCm Linux
    if torch.cuda.is_available():
        print("--> GPU Detected via CUDA/ROCm!")
        return torch.device("cuda")
        
    print("--> GPU not found. Falling back to system CPU.")
    return torch.device("cpu")

# 2. Local File Import Checks
try:
    from model import RawNet2
    # Ensure your dataset loaders can be accessed cleanly
    # Assuming you wrote them in dataset.py or separate files
    from dataset import RawNet2Dataset, ASVspoof2021EvalDataset
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Make sure your model.py and dataset.py are located in: ", os.getcwd())
    sys.exit(1)

def run_test_bench():
    # Initialize Device
    device = get_optimal_device()
    
    # ==========================================
    # STEP 1: DEFINE DATASET PATHS
    # ==========================================
    # Change these paths to point directly to your real storage folders
    # Hint: Use cd /d drive_letter: if your data is on an external HDD/SSD!
    PATH_TO_2019_TRAIN_AUDIO = r"H:\Downloads\rawnet 2 (thesis)\ASVspoof2019_LA_train\flac"
    PATH_TO_2019_TRAIN_PROTOCOL = r"H:\Downloads\rawnet 2 (thesis)\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
    
    PATH_TO_2021_EVAL_AUDIO = r"H:\Downloads\rawnet 2 (thesis)\ASVspoof2021_LA_eval\flac"
    PATH_TO_2021_EVAL_PROTOCOL = r"H:\Downloads\rawnet 2 (thesis)\ASVspoof2021_LA_eval\ASVspoof2021.LA.cm.eval.trl.txt"

    # ==========================================
    # STEP 2: INSTANTIATE DATA LOADERS
    # ==========================================
    print("\n--- Initializing Datasets ---")
    
    # Check if the directories actually exist before building
    if not os.path.exists(PATH_TO_2019_TRAIN_AUDIO):
        print(f"Warning: Training path '{PATH_TO_2019_TRAIN_AUDIO}' not found. Test bench running in dummy-mode.")
        return

    # A. 2019 Train Loader
    train_dataset = RawNet2Dataset(
        data_dir=PATH_TO_2019_TRAIN_AUDIO, 
        protocol_file=PATH_TO_2019_TRAIN_PROTOCOL
    )
    
    # --- ADD THIS DEBUG PRINT LINE ---
    print(f"DEBUG: train_dataset size is {len(train_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    # Using batch_size=8 or 16 is safe for an 8GB RX 6600
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

    # B. 2021 Eval Loader
    eval_dataset = ASVspoof2021EvalDataset(
        data_dir=PATH_TO_2021_EVAL_AUDIO, 
        protocol_file=PATH_TO_2021_EVAL_PROTOCOL
    )
    eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=0)

    # ==========================================
    # STEP 3: INITIALIZE RAWNET2 AND OPTIMIZER
    # ==========================================
    print("\n--- Initializing RawNet2 Model Architecture ---")
    model = RawNet2(out_classes=2).to(device)

    # --- ADD THIS WORKAROUND FOR DIRECTML ---
    if 'privateuseone' in str(device) or 'directml' in str(device):
        print("--> DirectML detected. Moving GRU layer parameters to CPU to bypass the fallback bug.")
        model.gru.cpu()
    # ----------------------------------------
    
    # Weight loss function to counteract class imbalance (if your dataset has way more fakes than real)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

    # Clear cache to free up fragmented VRAM
    if 'cuda' in str(device):
        torch.cuda.empty_cache()
    elif 'directml' in str(device):
        # If using torch_directml, it doesn't have an empty_cache() call,
        # but manual garbage collection can help reclaim host-side leaks:
        import gc
        gc.collect()

    # ==========================================
    # STEP 4: RUN ONE PASS SANITY TEST
    # ==========================================
    print("\n--- Executing 1-Batch Pipeline Sanity Check ---")
    model.train()
    epoch_loss = 0.0
    
    # Fetch a single batch of real data from the 2019 Train set
    for batch_idx, (audio, labels) in enumerate(train_loader):
        # Push batch tensors to DirectML or CUDA device
        audio = audio.to(device)
        labels = labels.to(device)
        
        # 2. Reset gradients
        optimizer.zero_grad()
        
        # 3. Forward Pass (No torch.no_grad() here so gradients can flow!)
        outputs = model(audio)
        loss = criterion(outputs, labels)
        
        # 4. Backward Pass and Optimization (Uncommented!)
        loss.backward()
        optimizer.step()
        
        # Track progress
        epoch_loss += loss.item()
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
    # Print average epoch loss
    avg_loss = epoch_loss / len(train_loader)
    print(f"\n[EPOCH COMPLETE] Average Training Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    run_test_bench()