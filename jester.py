import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torchvision import transforms

def get_img_filelist(video_folder):
    jpgs = sorted([f for f in os.listdir(video_folder) if f.endswith('.jpg')])
    return [os.path.join(video_folder, f) for f in jpgs]


def fix_frames_length(frames, desired_len=32):
    current_len = len(frames)
    if current_len == desired_len:
        return frames
    elif current_len > desired_len:
        # Több frame van, levágás az elejéről és végéről
        excess = current_len - desired_len
        cut_front = excess // 2
        cut_back = excess - cut_front
        return frames[cut_front:current_len - cut_back]
    else:
        # Kevesebb frame van, bővítés ismétléssel az egyik szélén
        deficit = desired_len - current_len
        
        # Számoljuk meg, mennyit ismételjünk az elején és végén
        repeat_front = deficit // 2
        repeat_back = deficit - repeat_front

        front_padding = [frames[0]] * repeat_front
        back_padding = [frames[-1]] * repeat_back
        return front_padding + frames + back_padding
    

# ---- Dataset object ----
class JesterDataset(Dataset):
    def __init__(self, root_dir: str, label_file: str, class_file: str, split: str = 'train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        annot_df = pd.read_csv(label_file, sep=';', header=None, names=['id', 'split', 'class_idx'])
        self.data = annot_df[annot_df['split'] == split].reset_index(drop=True)
        self.data['class_idx'] -= 1  # osztály index 0-tól
        class_df = pd.read_csv(class_file, sep=';', header=None, names=['class_idx', 'class_name'])
        self.class_idx_to_name = dict(zip(class_df['class_idx']-1, class_df['class_name']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_id = str(row['id'])
        class_idx = row['class_idx']
        class_name = self.class_idx_to_name[class_idx]

        vid_folder = os.path.join(self.root_dir, video_id)
        img_files = get_img_filelist(vid_folder)
        imgs = [Image.open(f).convert('RGB') for f in img_files]
        imgs = fix_frames_length(imgs)
        # Tipp: ha csak fix számú képre van szükség, vágd: imgs = imgs[:30]

        if self.transform is not None:
            imgs = [self.transform(img) for img in imgs]

        imgs = torch.stack(imgs)  # shape: (T, C, H, W) ha a transform tensorra alakít
        return imgs, int(class_idx), class_name

# ---- Példa ----

def main():
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])

    # ---- PARAMÉTEREK ----
    LABEL_FILE = '/Users/levente/Documents/SZTAKI/jester_dataset/jester-label.csv'
    CLASS_FILE = '/Users/levente/Documents/SZTAKI/jester_dataset/class-index.csv'
    JESTER_ROOT = '/Users/levente/Documents/SZTAKI/jester_dataset/jester'
    # Ide írd be a saját mappáid útvonalát, pl. '/media/data/jester'

    train_dataset = JesterDataset(JESTER_ROOT, LABEL_FILE, CLASS_FILE, split='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    # ---- Egy batch kipróbálása ----
    for imgs, class_idx, class_name in train_loader:
        print('frames tensor shape:', imgs.shape)
        print('class_idx:', class_idx)
        print('class_name:', class_name)
        break


if __name__ == '__main__':
    main()
