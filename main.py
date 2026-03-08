# %%
import torch
import torch.nn as nn
from torch import optim
from dataset import get_dataloader
from model import NGnet
from train import Trainer
import matplotlib.pyplot as plt
from psd import plot_psd_scatter
from statistic_of_errors import evaluate_psd
# %%
if __name__ == "__main__":
    # %%
    #Данные
    train_dataloader, test_dataloader, (X_val, y_val) = get_dataloader(number_of_samples = 20000, amp_range=(2, 15), length=200, noise_std = 1)
    # %%
    #Модель
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NGnet()
    model.to(device=device)
    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-4)
    # %%
    #Тренировка
    train_method = Trainer(
        model=model,
        loss=loss,
        optimizer=optimizer,
        device=device,
    )
    history_of_train = train_method.fit(
        test_dataloader=test_dataloader,
        train_dataloader=train_dataloader,
        num_epochs=10
    )
    # %%
    #Отрисовка
    plot_psd_scatter(X=X_val, start_time=20, tail_time=30, y=y_val)
    

    # %%
    min_errors, best_treshold = evaluate_psd(X_val, y_val)
