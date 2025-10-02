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

def test_model(model, test_loader, device):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            print(f"Test Batch {i+1}/{len(test_loader)}", end='\r')
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


def main():
    # Settings
    checkpoint_path = "checkpoints/best_model_epoch_20.pth"  # Adjust as needed
    batch_size = 2
    portion = 1.0

    # Model initialization
    model = GestureRecognizer()
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))
        print("Checkpoint loaded successfully.")
    else:
        print(f"No checkpoint found at {checkpoint_path}. Exiting.")
        return
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets and DataLoaders
    label_file = '/Users/levente/Documents/SZTAKI/jester_dataset/jester-label.csv'
    class_file = '/Users/levente/Documents/SZTAKI/jester_dataset/class-index.csv'
    jester_root = '/Users/levente/Documents/SZTAKI/jester_dataset/jester'

    test_dataset = JesterDataset(jester_root, label_file, class_file, split='test', portion=portion, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
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

    # Start testing
    test_model(model, test_loader, device)


if __name__ == '__main__':
    main()