import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from torchvision import transforms
from PIL import Image
from jester_detector import JesterDetectorDataset
from model_detector import GestureDetectorConv3d
import pandas as pd


def test_model(model, test_loader, device):
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct_val, total_val, test_loss = 0, 0, 0.0

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            count = (preds == labels).sum().item()
            print(f"Test {i+1}/{len(test_loader)} | Count: {count}/{len(labels)}", end='\r')
            correct_val += count
            total_val += len(labels)

    accuracy = correct_val / total_val if total_val > 0 else 0
    test_loss /= total_val if total_val > 0 else 0
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {accuracy * 100:.2f}%")


def main():
    # Settings
    checkpoint_path = "checkpoints/best_model_epoch_x.pth"
    batch_size = 32

    label_file = '../dataset/jester/test-split.csv'
    jester_root = '../dataset/jester'

    # Device configuration
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print("Using device:", device)

    # Model initialization
    model = GestureDetectorConv3d()
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded successfully.")
        print(f"Val loss: {checkpoint['val_loss']:.4f}, acc: {checkpoint['val_acc']:.4f}")
    else:
        print(f"No checkpoint found at {checkpoint_path}. Exiting.")
        return
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4494, 0.4263, 0.4075], std=[0.2692, 0.2721, 0.2705])
    ])

    # Datasets and DataLoaders
    df = pd.read_csv(label_file, sep=';')

    test_dataset = JesterDetectorDataset(jester_root, df, transform=transform, clip_len=10)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print("Dataset loaded")

    # Start testing
    test_model(model, test_loader, device)


if __name__ == '__main__':
    main()
