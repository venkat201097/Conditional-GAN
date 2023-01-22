# import argparse
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)
from tqdm import tqdm
# import numpy as np
import torch
from utils import get_mnist_data, generate

from GAN import D_net, G_net


def train(train_loader, D, G, num_epochs, batch_size, D_optimizer, G_optimizer, D_loss_fn, G_loss_fn, device):
    for epoch in range(num_epochs):
        with tqdm(enumerate(train_loader), total=len(train_loader), unit="batch", desc="Epoch {}".format(epoch)) as tepoch:
            for batch_idx, (x, y) in tepoch:

                # Data processing
                x = x.reshape(x.size(0), -1).to(device)
                true_labels = torch.ones(batch_size, device=device)
                false_labels = torch.zeros(batch_size, device=device)

                # Generator training
                G_optimizer.zero_grad()
                G.train()
                D.eval()
                g_out = G.generate(batch_size)
                d_g_out = D(g_out)
                g_loss = G_loss_fn(d_g_out, true_labels)
                g_loss.backward()
                G_optimizer.step()

                # Discriminator training
                D_optimizer.zero_grad()
                D.train()
                G.eval()
                d_real_out = D(x)
                d_fake_out = D(g_out.detach())
                d_loss = torch.mean(D_loss_fn(d_real_out, true_labels) + D_loss_fn(d_fake_out, false_labels))
                d_loss.backward()
                D_optimizer.step()
                
                # Progress bar update
                tepoch.set_postfix(GLoss=g_loss.item(), DLoss=d_loss.item())
    
    return G, D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 10
BATCH_SIZE = 100
train_loader, test_loader = get_mnist_data(BATCH_SIZE, device)

G = G_net(device=device).to(device)
D = D_net().to(device)

D_optimizer = torch.optim.Addam(
    D.parameters(),
    lr=1e-3
)
G_optimizer = torch.optim.Adam(
    G.parameters(),
    lr=1e-3
)
G_loss_fn = torch.nn.BCELoss()
D_loss_fn = torch.nn.BCELoss()

G, D = train(train_loader, D, G, NUM_EPOCHS, BATCH_SIZE, D_optimizer, G_optimizer, D_loss_fn, G_loss_fn, device)

generate(G, 200)

