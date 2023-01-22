import torch
from torch import nn
# import torch.nn.functional as F

# import logging

# bce_loss = nn.BCELoss()
class G_net(nn.Module):
    def __init__(self, z_dim=100):
        super(G_net, self).__init__()
        self.z_dim = z_dim

        self.net = nn.Sequential(
            nn.Linear(self.z_dim, 240),
            nn.ReLU(),
            nn.Linear(240, 240),
            nn.ReLU(),
            nn.Linear(240, 784),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.net(x)
        return x
    
    def generate(self, batch_size):
        z = torch.randn(batch_size, self.z_dim).to(device)
        x = self.net(z)
        return x


class D_net(nn.Module):
    def __init__(self):
        super(D_net, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(784, 240),
            nn.ReLU(),
            nn.Linear(240, 240),
            nn.ReLU(),
            nn.Linear(240, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.net(x)
        return x.squeeze(1)
    
    # def loss(self, x, labels):
    #     x = self.net(x)
    #     return bce_loss(x, labels)

# class GAN(nn.Module):
#     def __init__(self, z_dim=100):
#         super(GAN, self).__init__()

#         self.G = G(z_dim)
#         self.D = D()
    
#     def forward()

