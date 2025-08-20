import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import torchvision.models as models

# ---------------------------
# 1️⃣ Dataset
# ---------------------------
class NPYChunkedDataset(Dataset):
    def __init__(self, class_paths, chunk_size=32, transform=None, verbose=False):
        """
        class_paths: {'GBM': 'path/to/gbm', 'Astro': 'path/to/astro', ...}
        chunk_size: Her dosyadan kaç slice alınacak
        """
        self.file_list = []
        self.labels = []
        self.chunk_size = chunk_size
        self.transform = transform
        self.verbose = verbose

        self.label_map = {cls_name: idx for idx, cls_name in enumerate(class_paths.keys())}

        # NPY dosyalarını topla
        for cls_name, path in class_paths.items():
            for file_name in os.listdir(path):
                if file_name.endswith('.npy'):
                    self.file_list.append(os.path.join(path, file_name))
                    self.labels.append(self.label_map[cls_name])

        # Dosya başına chunk sayısını hesapla
        self.file_chunks = []
        if self.verbose:
            print("NPY dosya bilgileri yükleniyor...")
        for f in tqdm(self.file_list, desc="NPY dosyalar işleniyor"):
            data = np.load(f, mmap_mode='r')
            n_slices = data.shape[0]
            n_chunks = math.ceil(n_slices / self.chunk_size)
            self.file_chunks.append(n_chunks)
            if self.verbose:
                print(f"Dosya: {os.path.basename(f)}, Slice: {n_slices}, Chunk sayısı: {n_chunks}")

        # Index map oluştur
        self.index_map = []
        for file_idx, n_chunks in enumerate(self.file_chunks):
            for chunk_idx in range(n_chunks):
                self.index_map.append((file_idx, chunk_idx))

        if self.verbose:
            print(f"Toplam örnek (chunk) sayısı: {len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, chunk_idx = self.index_map[idx]
        file_path = self.file_list[file_idx]
        label = self.labels[file_idx]

        # Dosya yükleme (lazy)
        data = np.load(file_path, mmap_mode='r')

        start = chunk_idx * self.chunk_size
        end = min((chunk_idx+1)*self.chunk_size, data.shape[0])
        chunk = data[start:end]

        # Normalize ve float32
        chunk = chunk.astype(np.float32) / 255.0
        chunk = torch.tensor(chunk)

        # Grayscale -> 3 kanal
        if chunk.ndim == 2:  # (H,W)
            chunk = chunk.unsqueeze(0).repeat(3,1,1)  # (3,H,W)
        elif chunk.ndim == 3:  # (N,H,W)
            chunk = chunk.unsqueeze(1).repeat(1,3,1,1)  # (N,3,H,W)
        elif chunk.ndim == 4:  # (N,H,W,C)
            chunk = chunk.permute(0,3,1,2)  # (N,C,H,W)

        if self.transform:
            chunk = self.transform(chunk)

        return chunk, label

# ---------------------------
# 2️⃣ Dataset & DataLoader
# ---------------------------
class_paths = {
    "GBM": "data/islenen_dicomlar_gbm",
    "Astro": "data/islenen_dicomlar_astro",
    "Olio": "data/islenen_dicomlar_olio"
}

dataset = NPYChunkedDataset(class_paths, chunk_size=512, verbose=True)
loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

# ---------------------------
# 3️⃣ Model (ResNet18)
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ---------------------------
# 4️⃣ Training loop
# ---------------------------
num_epochs = 5


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (batch_imgs, batch_labels) in enumerate(pbar):
        batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)

        # Flatten: (batch_size, N_chunk, C,H,W) -> (N_total, C,H,W)
        if batch_imgs.ndim == 5:
            N = batch_imgs.shape[1]
            batch_imgs = batch_imgs.view(-1, batch_imgs.shape[2], batch_imgs.shape[3], batch_imgs.shape[4])
            batch_labels = batch_labels.repeat_interleave(N).to(device)  # slice başına label

        optimizer.zero_grad()
        outputs = model(batch_imgs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_imgs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_labels).sum().item()
        total += batch_labels.size(0)

        # tqdm bar altında batch loss ve epoch ortalama loss göster
        pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}", "Epoch Avg": f"{running_loss/total:.4f}"})

    # Epoch sonunda toplam loss ve accuracy
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

    # Modeli kaydet
    save_path = f"resnet18_epoch{epoch+1}_loss{epoch_loss:.4f}_acc{epoch_acc:.4f}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model kaydedildi: {save_path}")

    # GPU temizle
    torch.cuda.empty_cache()