import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from jester import JesterDataset



def calculate_dataset_statistics(jester_root, label_file, class_file, split=None, num_samples=None):
    """
    Kiszámolja a dataset mean és std értékeit RGB csatornánként.
    
    Args:
        jester_root: A Jester dataset root könyvtára
        label_file: A label CSV fájl path-ja
        class_file: A class names fájl path-ja
        split: 'train', 'val', vagy 'test'
        num_samples: Ha megadod, csak ennyi sample-t használ (gyorsabb teszteléshez)
    
    Returns:
        mean, std: Tuple a három RGB csatorna mean és std értékeivel
    """
    
    # Csak ToTensor transform, NINCS Normalize!
    # A képeket 0-1 tartományba kell hozni először
    basic_transform = transforms.Compose([
        transforms.ToTensor(),  # Konvertál 0-1 tartományba
    ])
    
    # Dataset létrehozása normalizálás NÉLKÜL
    dataset = JesterDataset(
        root_dir=jester_root,
        label_file=label_file,
        class_file=class_file,
        split=split,
        portion=1.0 if num_samples is None else num_samples / len(JesterDataset(root_dir=jester_root, label_file=label_file, class_file=class_file, split=split, portion=1.0, transform=basic_transform)),
        transform=basic_transform
    )
    
    print(f"Calculating statistics for {split} split...")
    print(f"Total videos: {len(dataset)}")
    
    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=1,  # Batch size = 1, mert videónként különböző frame számok lehetnek
        shuffle=False,
        num_workers=4
    )
    
    # Placeholders a sum és squared sum számára
    # Shape: (3,) - RGB csatornánként
    psum = torch.zeros(3)
    psum_sq = torch.zeros(3)
    total_pixels = 0
    
    print("Processing videos...")
    for videos, _, _ in tqdm(loader):
        # videos shape: [batch_size, num_frames, 3, H, W]
        # Például: [1, 22, 3, 224, 224]
        
        batch_size, num_frames, channels, height, width = videos.shape
        
        # Átrendezzük: [batch_size * num_frames, channels, height, width]
        videos = videos.view(batch_size * num_frames, channels, height, width)
        
        # Összegzés a batch, height, width dimenziókon (0, 2, 3), de MEGTARTJUK a channel dimenziót (1)
        psum += videos.sum(dim=[0, 2, 3])
        psum_sq += (videos ** 2).sum(dim=[0, 2, 3])
        
        # Pixelek száma: batch_size * num_frames * height * width
        total_pixels += batch_size * num_frames * height * width
    
    # Mean kiszámítása
    mean = psum / total_pixels
    
    # Std kiszámítása: std = sqrt(E[X^2] - (E[X])^2)
    std = torch.sqrt((psum_sq / total_pixels) - (mean ** 2))

    print(f"\n{split.upper() if split else 'Full'} Dataset Statistics:")
    print(f"Mean: {mean}")
    print(f"Std:  {std}")
    print(f"\nFormatted for transforms.Normalize():")
    print(f"mean=[{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}]")
    print(f"std=[{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]")
    
    return mean, std


def norm_std():
    # ÁLLÍTSD BE EZEKET A SAJÁT PATH-JAIDRA!
    JESTER_ROOT = "jester"  # A videók root könyvtára
    LABEL_FILE = "jester-label.csv"  # A label CSV fájl
    CLASS_FILE = "class-index.csv"  # A class names fájl

    # Teljes dataset statisztikák
    print("Calculating dataset statistics...\n")
    dataset_mean, dataset_std = calculate_dataset_statistics(
        jester_root=JESTER_ROOT,
        label_file=LABEL_FILE,
        class_file=CLASS_FILE,
        split=None,
        num_samples=None  # None = összes sample
    )

    # Dataset statisztikák
    print("Calculating dataset statistics...\n")
    train_mean, train_std = calculate_dataset_statistics(
        jester_root=JESTER_ROOT,
        label_file=LABEL_FILE,
        class_file=CLASS_FILE,
        split='train',
        num_samples=None  # None = összes sample
    )
    
    print("\n" + "="*60)
    
    print("Calculating VAL set statistics...\n")
    val_mean, val_std = calculate_dataset_statistics(
        jester_root=JESTER_ROOT,
        label_file=LABEL_FILE,
        class_file=CLASS_FILE,
        split='val',
        num_samples=None
    )
    
    print("\n" + "="*60)
    
    print("Calculating TEST set statistics...\n")
    test_mean, test_std = calculate_dataset_statistics(
        jester_root=JESTER_ROOT,
        label_file=LABEL_FILE,
        class_file=CLASS_FILE,
        split='test',
        num_samples=None
    )
    
    print("\n" + "="*60)
    print("\nCOMPARISON:")
    print(f"Full dataset mean: {dataset_mean}")
    print(f"Train mean: {train_mean}")
    print(f"Val mean:   {val_mean}")
    print(f"Test mean:  {test_mean}")
    print(f"\nFull dataset std:  {dataset_std}")
    print(f"Train std:  {train_std}")
    print(f"Val std:    {val_std}")
    print(f"Test std:   {test_std}")
    
    print(f"\nImageNet stats for comparison:")
    print(f"Mean: [0.485, 0.456, 0.406]")
    print(f"Std:  [0.229, 0.224, 0.225]")


def class_distribution():
    label_file = "jester-label.csv"
    val_df = pd.read_csv(label_file, sep=';', header=None, names=['id', 'split', 'class_idx'])
    val_counts = val_df[val_df['split'] == 'val']['class_idx'].value_counts().sort_index()
    test_counts = val_df[val_df['split'] == 'test']['class_idx'].value_counts().sort_index()

    comparison_df = pd.DataFrame({
        'Validation': val_counts,
        'Test': test_counts,
        'Difference': val_counts - test_counts,
    })

    print(comparison_df)


def mean_length():
    jester_root = "jester"  # A videók root könyvtára
    label_file = "jester-label.csv"  # A label CSV fájl

    df = pd.read_csv(label_file, sep=';', header=None, names=['id', 'split', 'class_idx'])
    total_frames = 0
    min_length = None
    min_index = None
    max_length = None
    max_index = None
    modus = {}

    print("Calculating mean video length...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        video_path = os.path.join(jester_root, str(row['id']))
        num_frames = sum(1 for f in os.listdir(video_path) if f.endswith('.jpg'))
        total_frames += num_frames
        if min_length is None or num_frames < min_length:
            min_length = num_frames
            min_index = row['id']
        if max_length is None or num_frames > max_length:
            max_length = num_frames
            max_index = row['id']
        if num_frames in modus:
            modus[num_frames] += 1
        else:
            modus[num_frames] = 1

    # Stats calculation
    mean_length = total_frames / len(df) if len(df) > 0 else 0
    print(f"Mean video length: {mean_length:.2f} frames")
    print(f"Min video length: {min_length} frames (at video ID: {min_index})")
    print(f"Max video length: {max_length} frames (at video ID: {max_index})")
    print(f"Modus video length: {max(modus, key=modus.get)} frames")
    #save modus to csv
    #modus_df = pd.DataFrame(list(modus.items()), columns=['Frame Count', 'Frequency'])
    #modus_df.to_csv('modus_distribution.csv', index=False)


if __name__ == "__main__":
    #norm_std()
    class_distribution()
    #mean_length()
