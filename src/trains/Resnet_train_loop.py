import sys
sys.path.append("src/utils")
sys.path.append("src/models")
sys.path.append("src/configs")

from dataloader import NPYChunkedDataset
from Resnet18_model import get_resnet18
import config


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset & Loader
dataset = NPYChunkedDataset(config.class_paths, chunk_size=config.chunk_size, verbose=True)
loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

# Model
model = get_resnet18(num_classes=config.num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.lr)

# Training loop
for epoch in range(config.num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
    for batch_imgs, batch_labels in pbar:
        batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)

        if batch_imgs.ndim == 5:
            N = batch_imgs.shape[1]
            batch_imgs = batch_imgs.view(-1, batch_imgs.shape[2], batch_imgs.shape[3], batch_imgs.shape[4])
            batch_labels = batch_labels.repeat_interleave(N).to(device)

        optimizer.zero_grad()
        outputs = model(batch_imgs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_imgs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_labels).sum().item()
        total += batch_labels.size(0)

        pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}", "Epoch Avg": f"{running_loss/total:.4f}"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{config.num_epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

    save_path = f"resnet18_epoch{epoch+1}_loss{epoch_loss:.4f}_acc{epoch_acc:.4f}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model kaydedildi: {save_path}")

    torch.cuda.empty_cache()
