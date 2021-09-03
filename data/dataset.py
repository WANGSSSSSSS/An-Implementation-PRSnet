import torch
from torch.utils.data.dataset import Dataset

class ShapeDataset(Dataset):
    def __init__(self):
        self.data_item = [1]*100
    def __getitem__(self, ind):
        batch = {}

        batch["voxel"] = torch.rand([1,32,32,32])
        batch["close_points"] = torch.rand(32*32*32,3)
        batch["sample"] = torch.rand(10,3)
        return batch

    def __len__(self):
        return len(self.data_item)