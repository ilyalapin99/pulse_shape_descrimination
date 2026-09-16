from torch.utils.data import Dataset
import torch

class PulseData(Dataset):
    def __init__(self, X, y, amplitudes):
        self.X = X
        self.y = y
        self.amplitudes = amplitudes

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):

        return torch.tensor(self.X[idx] / np.max(self.X[idx]), dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32), self.amplitudes[idx]   