import torch
from torch import nn

class G_net(nn.Module):
    def __init__(self, z_dim=100, device='cpu'):
        super(G_net, self).__init__()
        self.z_dim = z_dim
        self.device = device
        
        self.net = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.net(x)
        return x
    
    def generate(self, batch_size):
        z = torch.randn(batch_size, self.z_dim).to(self.device)
        x = self.net(z)
        return x


class D_net(nn.Module):
    def __init__(self):
        super(D_net, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.net(x)
        return x.squeeze(1)

