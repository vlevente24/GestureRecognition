import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from torchvision import transforms
from PIL import Image
from jester import JesterDataset
from model import GestureRecognizer
import time
import datetime
import argparse

# Példaként használjuk a GestureRecognizer modellt és két dataloader-t
# Feltételezzük, hogy:
# - model = GestureRecognizer() (definiálva külön)
# - train_loader, val_loader, test_loader (DataLoader példányok)
# - device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# - NUM_EPOCHS = 100, EARLY_STOPPING_PATIENCE = 10

def nice_time(time: float) -> str:
    return datetime.datetime.fromtimestamp(time).strftime('%H:%M:%S')


def delta_time(seconds: float) -> str:
    return str(datetime.timedelta(seconds=int(seconds)))


def train_model(model, train_loader, val_loader, test_loader, device, save_dir="checkpoints",
                num_epochs=100, early_stopping_patience=10, min_improve=1e-4, checkpoint_epoch=1):
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_val_loss = np.inf
    epochs_no_improve = 0

    # ha van checkpoint, előbb validálunk
    if checkpoint_epoch > 0:
        print(f"Checkpoint loaded, validating after epoch {checkpoint_epoch}...")
        model.eval()
        val_loss = 0.0
        correct_val, total_val = 0, 0
        #class_counts = torch.zeros(27, dtype=torch.int64).to(device)   # class ratios
        with torch.no_grad():
            for i, (X_val, y_val, _) in enumerate(val_loader):
                #class_counts += y_val.bincount(minlength=27).to(device)    # class ratios
                print(f"Validation: {i+1}/{len(val_loader)}", end='\r')
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item() * X_val.size(0)
                _, preds = outputs.max(1)
                correct_val += (preds == y_val).sum().item()
                total_val += y_val.size(0)
        val_loss /= total_val
        val_acc = correct_val / total_val
        best_val_loss = val_loss
        print(f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}")
    else:
        print("No checkpoint, training from scratch.")


    timestamp = time.time()
    for epoch in range(checkpoint_epoch+1, num_epochs+1):
        timestamp = time.time()
        print()
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        for i, (X, y, _) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
            print(f"Epoch {epoch}, Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}", end='\r')
            _, predicted = outputs.max(1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        print(f"\nTime: {nice_time(time.time())} | Epoch {epoch}/{num_epochs} completed in {delta_time(time.time() - timestamp)}.")

        # Időnként validálunk
        if epoch % 2 == 0:
            model.eval()
            val_loss = 0.0
            correct_val, total_val = 0, 0
            with torch.no_grad():
                for i, (X_val, y_val, _) in enumerate(val_loader):
                    print(f"Validation: {i+1}/{len(val_loader)}", end='\r')
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    outputs = model(X_val)
                    loss = criterion(outputs, y_val)
                    val_loss += loss.item() * X_val.size(0)
                    _, preds = outputs.max(1)
                    correct_val += (preds == y_val).sum().item()
                    total_val += y_val.size(0)
            val_loss /= total_val
            val_acc = correct_val / total_val

            print(f"Epoch {epoch}   Train loss: {train_loss:.4f} acc: {train_acc:.4f}  "
                  f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}")

            # Early stopping, modell mentés
            if val_loss < best_val_loss - min_improve:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_epoch_{epoch}.pth"))
                print(f" -- Model saved: {epoch}. epoch (Val loss improved: {best_val_loss:.4f})")
            else:
                epochs_no_improve += 1
                torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch}.pth"))
                print(f" -- Model saved: {epoch}. epoch (Val loss did not improve: {val_loss:.4f})")
                if epochs_no_improve >= early_stopping_patience:
                    print("Early stopping!")
                    break
        else:
            print(f"Epoch {epoch}   Train loss: {train_loss:.4f} acc: {train_acc:.4f}")

        print(f"Time: {nice_time(time.time())} | Epoch + Validation {epoch}/{num_epochs} completed in {delta_time(time.time() - timestamp)}.")

    # Test with the best model
    print("Testing the best model on the test set...")
    best_models = sorted([f for f in os.listdir(save_dir) if f.endswith('.pth')])
    model.load_state_dict(torch.load(os.path.join(save_dir, best_models[-1])))
    model.eval()
    test_loss, correct_test, total_test = 0.0, 0, 0
    with torch.no_grad():
        for X_test, y_test, _ in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            outputs = model(X_test)
            loss = criterion(outputs, y_test)
            test_loss += loss.item() * X_test.size(0)
            _, preds = outputs.max(1)
            correct_test += (preds == y_test).sum().item()
            total_test += y_test.size(0)
    test_loss /= total_test
    test_acc = correct_test / total_test
    print(f"Final test results - Loss: {test_loss:.4f}  Acc: {test_acc:.4f}")

# Usage example:
# train_model(model, train_loader, val_loader, test_loader, device)

def main():
    # Arguments
    parser = argparse.ArgumentParser(description="Training script for Gesture Recognition")
    parser.add_argument("--checkpoint", default=0, type=int, help="Checkpoint epoch to load. 0 means no checkpoint.")
    args = parser.parse_args()

    # Parameters
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    PORTION = 0.5  # use only a portion of the dataset for quicker experiments
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    CHECKPOINT_EPOCH = args.checkpoint  # if 0, no checkpoint is loaded

    # Model initialization
    model = GestureRecognizer()

    if CHECKPOINT_EPOCH > 0:
        chpt_best_path = f"checkpoints/best_model_epoch_{CHECKPOINT_EPOCH}.pth"
        chpt_path = f"checkpoints/model_epoch_{CHECKPOINT_EPOCH}.pth"
        if os.path.exists(chpt_best_path):
            model.load_state_dict(torch.load(chpt_best_path))
            print(f"Loaded checkpoint: {chpt_best_path}")
        elif os.path.exists(chpt_path):
            model.load_state_dict(torch.load(chpt_path))
            print(f"Loaded checkpoint: {chpt_path}")
        else:
            CHECKPOINT_EPOCH = 0
            print("No checkpoint loaded, starting from scratch.")
    else:
        print("No checkpoint loaded, starting from scratch.")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),   # random color jitter
        transforms.RandomRotation(degrees=10),  # random rotation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # ImageNet norm (if using ImageNet pretrained model)
    ])
    
    # Datasets and DataLoaders
    label_file = '/Users/levente/Documents/SZTAKI/jester_dataset/jester-label.csv'
    class_file = '/Users/levente/Documents/SZTAKI/jester_dataset/class-index.csv'
    jester_root = '/Users/levente/Documents/SZTAKI/jester_dataset/jester'

    train_dataset = JesterDataset(jester_root, label_file, class_file, split='train', portion=PORTION, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    val_dataset = JesterDataset(jester_root, label_file, class_file, split='val', portion=PORTION, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    test_dataset = JesterDataset(jester_root, label_file, class_file, split='test', portion=PORTION, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print("Dataset loaded")

    # Device configuration
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print("Using device:", device)

    # Start training
    train_model(model, train_loader, val_loader, test_loader, device, save_dir="checkpoints", num_epochs=EPOCHS, early_stopping_patience=EARLY_STOPPING_PATIENCE, checkpoint_epoch=CHECKPOINT_EPOCH)

if __name__ == '__main__':
    main()


# srun -pgpu_long --gres=gpu:v100:1 --pty bash -l
# srun -pgpu --gres=gpu:a100:1 --pty bash -l
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# chmod +x Miniconda3-latest-Linux-x86_64.sh
# ./Miniconda3-latest-Linux-x86_64.sh
# conda env create -f condaenv.yml
# ---------------------
# source ~/.bashrc
# tmux new -s train
# conda activate gestrec
# python train.py
# ---------------------
# teszt job:
# srun -pgpu -t 1:00:00 --gres=gpu:v100:1 --pty bash -l
# i=0;while true;do echo $i;i=$((i+1));sleep 2;done