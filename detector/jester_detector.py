import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torchvision import transforms


def get_img_filelist(video_folder):
    jpgs = sorted([f for f in os.listdir(video_folder) if f.endswith('.jpg')])
    return [os.path.join(video_folder, f) for f in jpgs]
    

def resample_list(data, n):
    idxs = np.linspace(0, len(data) - 1, n).round().astype(int)
    return [data[i] for i in idxs]


def class2bin(class_idx):
    if class_idx == 0 or class_idx == 2:
        return 0  # no gesture
    else:
        return 1  # gesture


# ---- Dataset object ----
class JesterDetectorDataset(Dataset):
    def __init__(self, root_dir: str, annot_df: pd.DataFrame, transform=None, clip_len=10):
        self.root_dir = root_dir
        self.transform = transform
        self.clip_len = clip_len
        self.data = annot_df
        self.data['class_idx'] -= 1  # osztály index 0-tól

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_id = str(row['id'])
        class_idx = row['class_idx']
        class_bin = class2bin(class_idx)

        vid_folder = os.path.join(self.root_dir, video_id)
        img_files = get_img_filelist(vid_folder)
        imgs = [Image.open(f).convert('RGB') for f in img_files]
        imgs = resample_list(imgs, self.clip_len)

        if self.transform is not None:
            imgs = [self.transform(img) for img in imgs]

        imgs = torch.stack(imgs)  # shape: (T, C, H, W) ha a transform tensorra alakít
        return imgs, class_bin



def main():
    print("JesterDetectorDataset példa")


if __name__ == '__main__':
    main()
