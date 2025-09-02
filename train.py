import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from torchvision import transforms
from PIL import Image
from jester import JesterDataset
from model import GestureRecognizer

# Példaként használjuk a GestureRecognizer modellt és két dataloader-t
# Feltételezzük, hogy:
# - model = GestureRecognizer() (definiálva külön)
# - train_loader, val_loader, test_loader (DataLoader példányok)
# - device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# - NUM_EPOCHS = 100, EARLY_STOPPING_PATIENCE = 10

def train_model(model, train_loader, val_loader, test_loader, device, save_dir="checkpoints",
                num_epochs=100, early_stopping_patience=10, min_improve=1e-4):
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_val_loss = np.inf
    epochs_no_improve = 0

    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        for i, (X, y, _) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
            print(f"Epoch {epoch}, Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}", end='\r')
            _, predicted = outputs.max(1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # Időnként validálunk
        if epoch % 2 == 0:
            model.eval()
            val_loss = 0.0
            correct_val, total_val = 0, 0
            with torch.no_grad():
                for X_val, y_val, _ in val_loader:
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
                torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_epoch{epoch}.pth"))
                print(f" -- Modell elmentve: {epoch}. epoch (Val loss javult: {best_val_loss:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print("Korai megállítás/Early stopping!")
                    break
        else:
            print(f"Epoch {epoch}   Train loss: {train_loss:.4f} acc: {train_acc:.4f}")

    # Teszt kiértékelés a megtanult modellel
    print("Legjobb modell betöltése értékeléshez...")
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
    print(f"Végső teszt eredmény - Loss: {test_loss:.4f}  Acc: {test_acc:.4f}")

# Használat:
# train_model(model, train_loader, val_loader, test_loader, device)

def main():
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])

    label_file = '/Users/levente/Documents/SZTAKI/jester_dataset/jester-label.csv'
    class_file = '/Users/levente/Documents/SZTAKI/jester_dataset/class-index.csv'
    jester_root = '/Users/levente/Documents/SZTAKI/jester_dataset/jester'

    train_dataset = JesterDataset(jester_root, label_file, class_file, split='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    val_dataset = JesterDataset(jester_root, label_file, class_file, split='val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    test_dataset = JesterDataset(jester_root, label_file, class_file, split='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    print("Dataset loaded")

    # Modell inicializálás
    model = GestureRecognizer()  # 27 osztály a Jester datasetben
    checkpoint_path = None # eg. "checkpoints/best_model_epochXX.pth" or None

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint loaded, starting from scratch.")

    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print("Használt eszköz/Device:", device)
    model.to(device)

    train_model(model, train_loader, val_loader, test_loader, device, save_dir="checkpoints", num_epochs=50, early_stopping_patience=10)

if __name__ == '__main__':
    main()
