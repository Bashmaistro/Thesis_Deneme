import os
import math
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

class NPYChunkedDataset(Dataset):
    def __init__(self, class_paths, chunk_size=32, transform=None, verbose=False):
        self.file_list = []
        self.labels = []
        self.chunk_size = chunk_size
        self.transform = transform
        self.verbose = verbose

        self.label_map = {cls_name: idx for idx, cls_name in enumerate(class_paths.keys())}

        for cls_name, path in class_paths.items():
            for file_name in os.listdir(path):
                if file_name.endswith('.npy'):
                    self.file_list.append(os.path.join(path, file_name))
                    self.labels.append(self.label_map[cls_name])

        self.file_chunks = []
        if self.verbose:
            print("NPY dosya bilgileri yükleniyor...")
        for f in tqdm(self.file_list, desc="NPY dosyalar işleniyor"):
            data = np.load(f, mmap_mode='r')
            n_slices = data.shape[0]
            n_chunks = math.ceil(n_slices / self.chunk_size)
            self.file_chunks.append(n_chunks)

        self.index_map = []
        for file_idx, n_chunks in enumerate(self.file_chunks):
            for chunk_idx in range(n_chunks):
                self.index_map.append((file_idx, chunk_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, chunk_idx = self.index_map[idx]
        file_path = self.file_list[file_idx]
        label = self.labels[file_idx]

        data = np.load(file_path, mmap_mode='r')
        start = chunk_idx * self.chunk_size
        end = min((chunk_idx+1)*self.chunk_size, data.shape[0])
        chunk = data[start:end]

        chunk = chunk.astype(np.float32) / 255.0
        chunk = torch.tensor(chunk)

        if chunk.ndim == 2:
            chunk = chunk.unsqueeze(0).repeat(3,1,1)
        elif chunk.ndim == 3:
            chunk = chunk.unsqueeze(1).repeat(1,3,1,1)
        elif chunk.ndim == 4:
            chunk = chunk.permute(0,3,1,2)

        if self.transform:
            chunk = self.transform(chunk)

        return chunk, label
