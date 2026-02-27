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

if __name__ == "__main__":
    test_pulse = torch.rand(64, 1, 128)
    model = NGnet()
    output = model(test_pulse)
    print(output.shape)