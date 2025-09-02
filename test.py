print("Legjobb modell betöltése értékeléshez...")
best_models = sorted([f for f in os.listdir(save_dir) if f.endswith('.pth')])
model.load_state_dict(torch.load(os.path.join(save_dir, best_models[-1])))
model.eval()
test_loss, correct_test, total_test = 0.0, 0, 0
with torch.no_grad():
    for X_test, y_test in test_loader:
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