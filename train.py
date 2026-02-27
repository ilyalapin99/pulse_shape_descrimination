import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np


class Trainer:
    def __init__(self, model, loss, optimizer, device, scheduler=None):
        
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
    
    def _calculate_metrics(self, targets, predictions):
        f1 = f1_score(targets, predictions, average='macro')
        unright_count = np.sum(predictions != targets)
        return f1, unright_count
    
    def _run_epoch(self, dataloader, is_training=True):
        
        if is_training:
            self.model.train()
        else:
            self.model.eval()
        
        context = torch.enable_grad() if is_training else torch.no_grad()

        all_preds, all_targets, all_loss = [], [], []

        with context:

            for X_batch, y_batch, _ in tqdm(dataloader):

                X_batch = X_batch.to(self.device).float()
                y_batch = y_batch.to(self.device).float()

                if is_training:
                    self.optimizer.zero_grad()

                output = self.model(X_batch)
                output = output.view(-1)
                loss = self.loss(output, y_batch)

                if is_training:
                    loss.backward()
                    self.optimizer.step()

                all_loss.append(loss.item())

                probs = torch.sigmoid(output)
                preds = (probs > 0.5).long()
                all_preds.append(preds.detach().cpu())
                all_targets.append(y_batch.detach().cpu())
            print(y_batch.shape)
            all_preds = torch.cat(all_preds).numpy()
            all_targets = torch.cat(all_targets).numpy()

            avg_loss = np.mean(all_loss)
            f1, un_right = self._calculate_metrics(all_targets, all_preds)
            return avg_loss, f1, un_right
        
    def fit(self, train_dataloader, test_dataloader, num_epochs):

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_f1": [],
            "test_f1": []
        }

        for epoch in range(num_epochs):

            print(f"Эпоха {epoch + 1}/{num_epochs}")

            train_loss, train_f1, train_errors = self._run_epoch(
                dataloader=train_dataloader,
                is_training=True
            )

            val_loss, val_f1, val_errors = self._run_epoch(
                dataloader=test_dataloader,
                is_training=False
            )
            history["train_loss"].append(train_loss)
            history['train_f1'].append(train_f1)
            history['val_loss'].append(val_loss)
            history['test_f1'].append(val_f1)

            print(f"TRAIN -> loss: {train_loss}, f1: {train_f1}")
            print(f"VALIDATION -> loss: {val_loss}, f1: {val_f1}, errors: {val_errors}")
        return history

if __name__ == "__main__":
    a = []
    a.append([1, 2])
    a.append([2, 3])
    mass = [torch.tensor(tens) for tens in a]
    print(torch.cat(mass))

