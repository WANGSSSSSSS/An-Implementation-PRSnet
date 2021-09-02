import torch
from torch.utils.data.dataset import Dataset

class ShapeDataset(Dataset):
    def __init__(self):
        self.data_item = []
    def __getitem__(self, ind):
        batch = {}

        batch["voxel"]
        batch["close_points"]
        batch["sample"]

        return batch

    def __len__(self):
        return len(self.data_item)