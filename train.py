import os
import sys
import gc
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.adam import Adam

# Import local modules
from model import RawNet2
from dataset import RawNet2Dataset


def get_optimal_device():
    try:
        import torch_directml
        if torch_directml.is_available():
            print("--> AMD GPU Detected via DirectML!")
            return torch_directml.device()
    except ImportError:
        pass
    
    if torch.cuda.is_available():
        print("--> GPU Detected via CUDA/ROCm!")
        return torch.device("cuda")
        
    print("--> GPU not found. Falling back to system CPU.")
    return torch.device("cpu")


def train_model():
    device = get_optimal_device()
    
    # 1. Dataset Paths
    PATH_TO_2019_TRAIN_AUDIO = r"H:\Downloads\rawnet 2 (thesis)\ASVspoof2019_LA_train\flac"
    PATH_TO_2019_TRAIN_PROTOCOL = r"H:\Downloads\rawnet 2 (thesis)\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"

    if not os.path.exists(PATH_TO_2019_TRAIN_AUDIO):
        print(f"Error: Path '{PATH_TO_2019_TRAIN_AUDIO}' not found.")
        sys.exit(1)

    # 2. Data Loader
    print("\n--- Initializing Training Dataset ---")
    train_dataset = RawNet2Dataset(
        data_dir=PATH_TO_2019_TRAIN_AUDIO, 
        protocol_file=PATH_TO_2019_TRAIN_PROTOCOL
    )
    
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 3. Model Setup & DirectML Fix
    print("\n--- Initializing RawNet2 Model Architecture ---")
    model = RawNet2(out_classes=2).to(device)
    
    if 'privateuseone' in str(device) or 'directml' in str(device):
        print("--> DirectML detected. Moving GRU layer parameters to CPU to bypass fallback bug.")
        model.gru.cpu()

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

    # 4. Checkpoint Directory Setup
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # =========================================================================
    # AUTO-RESUME CHECK: Look for the latest saved checkpoint
    # =========================================================================
    start_epoch = 1
    existing_checkpoints = glob.glob(os.path.join(checkpoint_dir, "rawnet2_epoch_*.pt"))
    
    if existing_checkpoints:
        # Extract the highest epoch number found in the folder
        checkpoint_epochs = [int(f.split("_")[-1].replace(".pt", "")) for f in existing_checkpoints]
        latest_epoch = max(checkpoint_epochs)
        latest_checkpoint_path = os.path.join(checkpoint_dir, f"rawnet2_epoch_{latest_epoch}.pt")
        
        print(f"\n[AUTO-RESUME] Found existing checkpoint: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path)
        
        # Load saved weights into the model and optimizer
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Start at the next epoch
        start_epoch = checkpoint['epoch'] + 1
        print(f"--> Successfully restored state. Resuming training from EPOCH {start_epoch}!")
    else:
        print("\n--> No previous checkpoints found. Starting fresh training from Epoch 1.")
    # =========================================================================

    # 5. Full Training Loop
    total_epochs = 20
    accumulation_steps = 2
    
    print(f"\n--- Starting Training (Epochs {start_epoch} to {total_epochs}) ---")
    
    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (audio, labels) in enumerate(train_loader):
            audio = audio.to(device)
            labels = labels.to(device)
            
            outputs = model(audio)
            loss = criterion(outputs, labels) / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * accumulation_steps
            
            if (batch_idx + 1) % 50 == 0:
                gc.collect()

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
                current_batch_loss = loss.item() * accumulation_steps
                print(f"Epoch [{epoch}/{total_epochs}] | Batch [{batch_idx + 1}/{len(train_loader)}] | Current Loss: {current_batch_loss:.4f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\n>>> [EPOCH {epoch} COMPLETE] Average Loss: {avg_epoch_loss:.4f} <<<")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"rawnet2_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }, checkpoint_path)
        print(f"--> Saved checkpoint to: {checkpoint_path}\n")

if __name__ == "__main__":
    train_model()