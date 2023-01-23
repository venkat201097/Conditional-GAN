# import argparse
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)
from tqdm import tqdm
# import numpy as np
import torch
from utils import get_mnist_data, generate, save_image, save_gif
import matplotlib.pyplot as plt

from cGAN import D_net, G_net

def train_D(D, real_data, generated_fake_data, true_labels, false_labels, D_optimizer, D_loss_fn):
    
    # Discriminator training
    D_optimizer.zero_grad()
    d_real_out = D(real_data)# real_labels)
    d_fake_out = D(generated_fake_data)#, real_labels)
    d_loss = D_loss_fn(d_real_out, true_labels) + D_loss_fn(d_fake_out, false_labels)
    d_loss.backward()
    D_optimizer.step()
    D_optimizer.zero_grad()
    return d_loss.item()

def train_G(G, D, batch_size, true_labels, G_optimizer, G_loss_fn):
    
    # Generator training
    G_optimizer.zero_grad()
    g_out = G.generate(batch_size, y)
    d_g_out = D(g_out)#, y)
    g_loss = G_loss_fn(d_g_out, true_labels)
    g_loss.backward()
    G_optimizer.step()
    G_optimizer.zero_grad()
    return g_loss.item()

def train(train_loader, D, G, num_epochs, batch_size, D_optimizer, G_optimizer, D_loss_fn, G_loss_fn, device):
    g_losses = []
    d_losses = []

    gif_z = torch.randn(200, 100).to(device)
    gif_y = torch.nn.functional.one_hot(torch.Tensor([int(i//20) for i in range(200)]).type(torch.LongTensor)).type(torch.FloatTensor).to(device)

    for epoch in range(num_epochs):
        epoch_d_loss = 0
        epoch_g_loss = 0

        with tqdm(enumerate(train_loader), total=len(train_loader), unit="batch", ascii=True, desc="Epoch {}".format(epoch)) as tepoch:
            for bid, (x, y) in tepoch:
                
                # Data processing
                x = x.reshape(x.size(0), -1).to(device)
                y = torch.nn.functional.one_hot(y, num_classes=10).type(torch.FloatTensor).to(device)

                true_labels = torch.ones(batch_size, device=device)
                false_labels = torch.zeros(batch_size, device=device)

                # Discriminator training
                D.train()
                G.eval()
                d_loss = train_D(D, x, G.generate(batch_size, y).detach(), true_labels, false_labels, D_optimizer, D_loss_fn)
                epoch_d_loss += d_loss

                # Generator training
                G.train()
                D.eval()
                g_loss = train_G(G, D, batch_size, y, true_labels, G_optimizer, G_loss_fn)
                epoch_g_loss += g_loss

                if epoch%10==0:
                    torch.save(D, 'SavedModels/D_epoch-{}'.format(epoch))
                    torch.save(G, 'SavedModels/G_epoch-{}'.format(epoch))
                
                # Progress bar update
                tepoch.set_postfix(GLoss=g_loss, DLoss=d_loss)
            tepoch.set_postfix(GLoss=epoch_g_loss/bid, DLoss=epoch_d_loss/bid)
            g_losses.append(epoch_g_loss/bid)
            d_losses.append(epoch_d_loss/bid)
        
        G.eval()
        epoch_image = G.generate(200, torch.nn.functional.one_hot(torch.randint(0, 10, (200, )), num_classes=10).type(torch.FloatTensor).to(device)).cpu().detach().view(-1,1,28,28)
        save_image(epoch_image, 'Image_Epoch-{}.jpg'.format(epoch))

        gif_image = G(gif_z, gif_y).cpu().detach().view(-1,1,28,28)
        save_image(gif_image, 'Gif/Gif_Epoch-{}.jpg'.format(epoch))
    
    torch.save(D, 'SavedModels/D_epoch-{}'.format(epoch))
    torch.save(G, 'SavedModels/G_epoch-{}'.format(epoch))
    
    plt.figure()
    plt.plot(g_losses, label='Generator loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.legend()
    plt.savefig('loss.png')

    return G, D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 100
BATCH_SIZE = 100
train_loader, test_loader = get_mnist_data(BATCH_SIZE, device)

G = G_net(device=device).to(device)
D = D_net().to(device)

D_optimizer = torch.optim.Adam(
    D.parameters(),
    lr=1e-4
)
G_optimizer = torch.optim.Adam(
    G.parameters(),
    lr=1e-4
)
G_loss_fn = torch.nn.BCELoss()
D_loss_fn = torch.nn.BCELoss()

G, D = train(train_loader, D, G, NUM_EPOCHS, BATCH_SIZE, D_optimizer, G_optimizer, D_loss_fn, G_loss_fn, device)

# generate(G, 200)
