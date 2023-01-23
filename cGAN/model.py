import torch
from torch import nn

class G_net(nn.Module):
    def __init__(self, z_dim=100, y_dim=10, device='cpu'):
        super(G_net, self).__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.device = device
        
        self.z_proj = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.LeakyReLU(0.2)
        )
        self.y_proj = nn.Sequential(
            nn.Linear(self.y_dim, 256),
            nn.LeakyReLU(0.2)
        )
        self.net = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, x, y):
        assert x.shape[0]==y.shape[0]
        x = self.z_proj(x)
        y = self.y_proj(y)
        x = torch.cat((x, y), dim=1)
        x = self.net(x)
        return x
    
    def generate(self, batch_size, y=None):
        z = torch.randn(batch_size, self.z_dim).to(self.device)
        x = self.z_proj(z)
        if y==None:
            y = torch.nn.functional.one_hot(torch.randint(0, 10, (batch_size, ))).type(torch.FloatTensor).to(self.device)
        y = self.y_proj(y)
        x = torch.cat((x, y), dim=1)
        x = self.net(x)
        return x


class D_net(nn.Module):
    def __init__(self, x_dim=784, y_dim=10):
        super(D_net, self).__init__()
        self.x_dim = x_dim
        #self.y_dim = y_dim
        
        self.x_proj = nn.Sequential(
            nn.Linear(self.x_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        '''self.y_proj = nn.Sequential(
            nn.Linear(self.y_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )'''
        self.net = nn.Sequential(
            #nn.Linear(1280, 1024),
            #nn.LeakyReLU(0.2),
            #nn.Dropout(0.3),
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
        #assert x.shape[0]==y.shape[0]
        x = self.x_proj(x)
        #y = self.y_proj(y)
        #x = torch.cat((x, y), dim=1)
        x = self.net(x)
        return x.squeeze(1)

