import torch
from tqdm import tqdm

class Trainer:
    def __init__(
            self,
            model, 
            criterion,
            optimizer,
            device,
            num_epochs,
            train_dataloader, 
            test_dataloader,
            warmup_beta,
            max_beta, 
            scheduler=None):
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.sheduler = scheduler
        self.num_epochs = num_epochs
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.warmup_beta = warmup_beta
        self.max_beta = max_beta


    def _run_epoch(self, dataloader, is_training=True, beta=1.0):
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        epoch_total_loss = 0.0
        epoch_reconstruction_loss = 0.0
        epoch_kl_loss = 0.0       

        context = torch.enable_grad() if is_training else torch.no_grad()

        with context:
            for batch in tqdm(dataloader):
                inputs = batch[0].to(self.device)

                if is_training:
                    self.optimizer.zero_grad()

                outputs, mu, logvar = self.model(inputs)
                total_loss, recon_loss, kl_loss = self.criterion(outputs, inputs.float().unsqueeze(1), mu, logvar, beta=beta)

                if is_training:
                    total_loss.backward()
                    self.optimizer.step()

                epoch_total_loss += total_loss.item()
                epoch_reconstruction_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()

        epoch_total_loss /= len(dataloader)
        epoch_reconstruction_loss /= len(dataloader)
        epoch_kl_loss /= len(dataloader)

        return epoch_total_loss, epoch_reconstruction_loss, epoch_kl_loss

    def fit(self):

        history = {
            "train_loss": [],
            "test_loss": [],
            "train_recon_loss": [],
            "test_recon_loss": [],
            "train_kl_loss": [],
            "test_kl_loss": [],
        }
        best_test_loss = float('inf')

        for epoch in range(self.num_epochs):
            beta = min(self.max_beta, self.max_beta * (epoch + 1) / self.warmup_beta)
            print(f"Epoch -> {epoch + 1}/{self.num_epochs}")

            train_loss, train_recon_loss, train_kl_loss = self._run_epoch(
                dataloader=self.train_dataloader,
                beta=beta,
                is_training=True
            )
            test_loss, test_recon_loss, test_kl_loss = self._run_epoch(
                dataloader=self.test_dataloader,
                beta=beta,
                is_training=False
            )

            if best_test_loss > test_loss and (epoch + 1) >= self.warmup_beta:
                best_test_loss = test_loss
                torch.save(self.model.state_dict(), 'checkpoints/best_vae_model.pth')

            print(f"TRAIN loss-> {train_loss}")
            print(f"TEST loss -> {test_loss}")

            #append in history
            history["train_loss"].append(train_loss)
            history["train_kl_loss"].append(train_kl_loss)
            history['train_recon_loss'].append(train_recon_loss)

            history["test_loss"].append(test_loss)
            history["test_kl_loss"].append(test_kl_loss)
            history['test_recon_loss'].append(test_recon_loss)

        return history


            

                
                


                






