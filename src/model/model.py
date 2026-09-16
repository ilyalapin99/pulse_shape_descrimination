import torch.nn as nn
import torch

class NGnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=2),
            nn.ReLU(),
            
        )
        self.fc = nn.Sequential(
            nn.Linear(1616, 1),
        )

        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = x.view(x.shape[0], -1)
        output = self.fc(x)

        return output
    
class PulseAutoEncoder(nn.Module):

    def __init__(self, out_channels_first=8, latent_dim=1,):
        super().__init__()
        self.out_channels_first = out_channels_first
        self.encoder = nn.Sequential(

            #[64, 1, 200]
            nn.Conv1d(in_channels=1, out_channels=out_channels_first, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(out_channels_first),
            nn.ReLU(),

            #[64, 8, 100]
            nn.Conv1d(in_channels=out_channels_first, out_channels=out_channels_first*2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(out_channels_first * 2),
            nn.ReLU(),

            # Сплющиваю
            nn.Flatten(),
            nn.Linear(out_channels_first * 2 * 50, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
            )
        
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels_first * 2 * 50)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(out_channels_first*2, out_channels_first, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(out_channels_first),
            nn.ReLU(),

            nn.ConvTranspose1d(out_channels_first, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(1),
            nn.Sigmoid(),

        )
        
    def forward(self, x):
        
        x = x.unsqueeze(1)
        latent = self.encoder(x)
        x_recon = self.decoder_linear(latent)

        x_recon = x_recon.view(-1, self.out_channels_first * 2, 50)
        x_recon = self.decoder(x_recon)
        
        return x_recon, latent

if __name__ == "__main__":
    test_pulse = torch.rand(64, 1, 128)
    model = NGnet()
    output = model(test_pulse)
    print(output.shape)

