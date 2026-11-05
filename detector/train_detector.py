import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from torchvision import transforms
from PIL import Image
from jester_detector import JesterDetectorDataset
from model_detector import GestureDetectorConv3d
import time
import datetime
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def nice_time(time: float) -> str:
    return datetime.datetime.fromtimestamp(time).strftime('%H:%M:%S')


def delta_time(seconds: float) -> str:
    return str(datetime.timedelta(seconds=int(seconds)))


def binary_class_weights(label_file):
    df = pd.read_csv(label_file, sep=';', header=None, names=['id', 'split', 'class_idx'])
    # Create binary label
    df['binary_class'] = df['class_idx'].apply(lambda x: 0 if x in [1, 3] else 1)
    counts = df['binary_class'].value_counts().sort_index()  # [0, 1]
    total = counts.sum()
    weights = total / (2.0 * counts)
    return torch.tensor(weights.values, dtype=torch.float32)


def train_model(model, train_loader, val_loader, device, save_dir="checkpoints",
                num_epochs=100, early_stopping_patience=10, min_improve=1e-4, checkpoint=None, class_weights=None):
    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    best_val_loss = None
    scheduler = None
    epochs_no_improve = 0

    # Checkpoint loading
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        checkpoint_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['val_loss']
        checkpoint_validation_accuracy = checkpoint['val_acc']
        epochs_no_improve = checkpoint['epochs_no_improve']

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6, last_epoch=checkpoint_epoch)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded checkpoint: {checkpoint_epoch}")
        print(f"Val loss: {best_val_loss:.4f} acc: {checkpoint_validation_accuracy:.4f}")
    else:
        checkpoint_epoch = 0
        best_val_loss = np.inf
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        print("No checkpoint loaded, starting from scratch.")


    timestamp = time.time()
    for epoch in range(checkpoint_epoch+1, num_epochs+1):
        timestamp = time.time()
        print()
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        for i, (X, y) in enumerate(train_loader):
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

        # Occasionally validate
        if epoch % 1 == 0:
            model.eval()
            val_loss = 0.0
            correct_val, total_val = 0, 0
            with torch.no_grad():
                for i, (X_val, y_val) in enumerate(val_loader):
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

            # Early stopping, saving model
            if val_loss < best_val_loss - min_improve:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'epochs_no_improve': epochs_no_improve
                }, os.path.join(save_dir, f"best_model_epoch_{epoch}.pth"))
                print(f" -- Model saved: {epoch}. epoch (Val loss improved: {best_val_loss:.4f})")
            else:
                epochs_no_improve += 1
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'epochs_no_improve': epochs_no_improve
                }, os.path.join(save_dir, f"model_epoch_{epoch}.pth"))
                print(f" -- Model saved: {epoch}. epoch (Val loss did not improve: {val_loss:.4f})")
                if epochs_no_improve >= early_stopping_patience:
                    print("Early stopping!")
                    break
        else:
            print(f"Epoch {epoch}   Train loss: {train_loss:.4f} acc: {train_acc:.4f}")

        scheduler.step()
        print(f"Time: {nice_time(time.time())} | Epoch + Validation {epoch}/{num_epochs} completed in {delta_time(time.time() - timestamp)}.")




def main():
    # Arguments
    parser = argparse.ArgumentParser(description="Training script for Gesture Detector")
    parser.add_argument("--checkpoint", default=None, type=str, help="Checkpoint path to load. If not provided, training starts from scratch.")
    args = parser.parse_args()

    # Check if the checkpoint exists
    if args.checkpoint is not None and not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file '{args.checkpoint}' not found.")

    # Parameters
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    PORTION = 1.0  # use only a portion of the dataset for quicker experiments
    EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 10
    CLIP_LEN = 20
    CHECKPOINT = args.checkpoint  # if 0, no checkpoint is loaded

    # Model initialization
    model = GestureDetectorConv3d()

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),   # random color jitter
        transforms.RandomRotation(degrees=10),  # random rotation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4494, 0.4263, 0.4075], std=[0.2692, 0.2721, 0.2705])   # Jester norm
    ])
    transformV = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4494, 0.4263, 0.4075], std=[0.2692, 0.2721, 0.2705])   # Jester norm
    ])
    
    # Datasets and DataLoaders
    label_file = '../dataset/jester-label.csv'
    jester_root = '../dataset/jester'

    # Drop the rest of the dataset (portion)
    df = pd.read_csv(label_file, sep=';', header=None, names=['id', 'split', 'class_idx'])
    if 0 < PORTION < 1.0:
        small_df, _ = train_test_split(df, train_size=PORTION, stratify=df['class_idx'], random_state=42)
        df = small_df
        
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)

    train_dataset = JesterDetectorDataset(jester_root, train_df, transform=transform, clip_len=CLIP_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    val_dataset = JesterDetectorDataset(jester_root, val_df, transform=transformV, clip_len=CLIP_LEN)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Class weights
    class_weights = binary_class_weights(label_file)

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
    train_model(model,
                train_loader,
                val_loader,
                device,
                save_dir="checkpoints",
                num_epochs=EPOCHS,
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                checkpoint=CHECKPOINT,
                class_weights=class_weights
                )

if __name__ == '__main__':
    main()
