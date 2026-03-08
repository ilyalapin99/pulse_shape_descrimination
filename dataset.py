import torch
from dataloader import generate_data
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

class PulseDataset(Dataset):
    def __init__(self, X, y, amplitudes, transform=True):
        self.X = X
        self.y = y
        self.amplitudes = amplitudes
        self.transform = transform
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = x / np.max(x)
        return x, self.y[idx], self.amplitudes[idx]

def get_dataloader(number_of_samples, amp_range, length, noise_std):
    
    #Загружаем данные
    X, y, amplitudes = generate_data(number_of_samples, amp_range, length, noise_std)
    X_train, X_test, y_train, y_test, amplitudes_train, amplitudes_test = train_test_split(
        X, y, amplitudes,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    """show_statictics(X_test, y_test, amplitudes_test, label='test')
    show_statictics(X_train, y_train, amplitudes_train, label='train')"""
    
    train_dataset = PulseDataset(X_train, y_train, amplitudes_train)
    test_dataset = PulseDataset(X_test, y_test, amplitudes_test)

    #Трейн Даталоадер
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader, (X_test, y_test)


def show_statictics(X, y, amp, label):
    plt.hist(amp, bins=30)
    plt.title(label)
    plt.show()
    print(f"{label} gamma count: {np.sum(y == 1)}")
    print(f"{label} neutron count: {np.sum(y == 0)}")

if __name__ == "__main__":
    train_loader, test_loader = get_dataloader(number_of_samples = 20000, amp_range=(7, 25), length=200, noise_std=1)
    X_batch, y_batch, _ = next(iter(train_loader))
    print(y_batch.shape)


    




    
